"""Turn the engine assets of a gufo-org/gufo release into linked .oe bundles.

Each asset named gufo-vX.Y.Z-OS-ARCH.EXT becomes gufo-vX.Y.Z-OS-ARCH.oe, a zip holding
only an openengine.json whose "download" field points at the original asset. Dots and
underscores in the asset name become hyphens, because bundle names may only contain
lowercase letters, numbers, and hyphens.

The manifest is filled from three sources:
- the asset name (OS, architecture, version),
- the archive itself (server path, bundled ROCm and HIP versions from amdhip64's version
  resource, kernel GPU targets, and glibc and kernel minimums for Linux binaries),
- upstream files at the release tag (HIP architectures from CMakeLists.txt, the supported
  GGUF models from docs/models/README.md, and the Windows baseline that
  tools/windows/check_bundle.py validates against).

Usage:
    python make_oe.py [--repo gufo-org/gufo] [--tag vX.Y.Z] [--out dist]

Requires Python 3.9+. No third-party packages. Set GITHUB_TOKEN or GH_TOKEN to avoid
API rate limits.
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ASSET_RE = re.compile(r"^gufo-v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)-(?P<os>windows|linux|macos)-"
                      r"(?P<arch>x86_64|amd64|aarch64|arm64)\.(?:zip|tar\.gz|tgz|tar\.xz)$")
ARCHIVE_RE = re.compile(r"\.(zip|tar\.gz|tgz|tar\.xz)$")
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
GFX_RE = re.compile(r"gfx[0-9a-f]+")
GLIBC_RE = re.compile(rb"GLIBC_(2\.[0-9]+(?:\.[0-9]+)?)\0")
HIP_DLL_RE = re.compile(r"^amdhip64(_[0-9]+)?\.dll$", re.IGNORECASE)
ZIP_DATE = (2000, 1, 1, 0, 0, 0)

ARCHES = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
OS_LABELS = {"windows": "Windows", "linux": "Linux", "macos": "macOS"}
WINDOWS_BUILDS = {"10": "10.0", "11": "10.0.22000"}
SERVER = "gufo"
API = ["openai-chat", "openai-completions", "openai-responses", "anthropic-messages"]
DEVICE = "AMD Strix Halo (Ryzen AI Max with Radeon 8060S)"


def github(url, raw=False):
    headers = {"Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
               "User-Agent": "make-oe"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return r.read().decode("utf-8") if raw else json.load(r)


def upstream_file(repo, path, tag):
    try:
        return github(f"https://api.github.com/repos/{repo}/contents/{path}?ref={tag}", raw=True)
    except urllib.error.HTTPError as e:
        print(f"warning: cannot read {path} at {tag}: {e}", file=sys.stderr)
        return ""


def find_release(repo, tag):
    """Return the release for tag, or the newest published release with an engine asset.

    Upstream attaches the Windows ZIP to a release after creating it, and older releases
    have no assets, so the newest release is not always usable.
    """
    if tag:
        return github(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
    for release in github(f"https://api.github.com/repos/{repo}/releases?per_page=30"):
        if release["draft"] or release["prerelease"]:
            continue
        if any(ASSET_RE.match(a["name"]) for a in release["assets"]):
            return release
        print(f"note: {release['tag_name']} has no engine assets yet", file=sys.stderr)
    raise SystemExit(f"no release with engine assets found in {repo}")


def tag_revision(repo, tag):
    ref = github(f"https://api.github.com/repos/{repo}/git/ref/tags/{tag}")["object"]
    if ref["type"] == "tag":
        ref = github(ref["url"])["object"]
    return ref["sha"]


def hip_architectures(cmake):
    match = re.search(r"set\(CMAKE_HIP_ARCHITECTURES\s+([^\s)]+)", cmake)
    return match.group(1).strip('"').split(";") if match else []


def windows_baseline(checker):
    """Return the minimum Windows build whose inbox DLLs upstream validates the bundle against."""
    match = re.search(r"Windows (1[01]) inbox", checker)
    return WINDOWS_BUILDS[match.group(1)] if match else None


def gguf_models(readme):
    """Return (name, weights) for the rows of docs/models/README.md that load GGUF files."""
    models = []
    for line in readme.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and "GGUF" in cells[1] and cells[0] not in ("Model", "---"):
            models.append((cells[0], cells[1]))
    return models


def pe_version(data):
    """Return the file version and description of a PE file's version resource."""
    pos = data.find(struct.pack("<I", 0xFEEF04BD))
    if pos < 0:
        return None, None
    ms, ls = struct.unpack_from("<II", data, pos + 8)
    version = (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)
    text = data.decode("utf-16-le", errors="ignore")
    match = re.search(r"FileDescription\x00+([^\x00]+)", text)
    return version, match.group(1) if match else None


def elf_notes(data):
    """Return the minimum Linux kernel (GNU ABI tag) of an ELF file."""
    out = {}
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return out
    end = "<" if data[5] == 1 else ">"
    phoff, = struct.unpack_from(end + "Q", data, 0x20)
    phentsize, phnum = struct.unpack_from(end + "HH", data, 0x36)
    for i in range(phnum):
        p_type, _, p_offset, _, _, p_filesz = struct.unpack_from(end + "IIQQQQ", data,
                                                                 phoff + i * phentsize)
        if p_type != 4:
            continue
        pos, stop = p_offset, p_offset + p_filesz
        while pos + 12 <= stop:
            namesz, descsz, ntype = struct.unpack_from(end + "III", data, pos)
            name = data[pos + 12:pos + 12 + namesz].rstrip(b"\0")
            desc = pos + 12 + ((namesz + 3) & ~3)
            pos = desc + ((descsz + 3) & ~3)
            if ntype == 1 and name == b"GNU" and descsz >= 16:
                out["kernel"] = struct.unpack_from(end + "IIII", data, desc)[1:]
    return out


def version_text(parts, keep=2):
    parts = list(parts)
    while len(parts) > keep and parts[-1] == 0:
        parts.pop()
    return ".".join(str(n) for n in parts)


def strip_wrapper(files):
    """Return paths as a runner sees them in bin/: without one shared top-level folder."""
    files = [re.sub(r"^(\./)+", "", f.replace("\\", "/")) for f in files]
    tops = {f.split("/", 1)[0] for f in files}
    if len(tops) == 1 and all("/" in f for f in files):
        return [f.split("/", 1)[1] for f in files]
    return files


def inspect(archive, name):
    """Read the facts a manifest needs from the archive's file list and binaries."""
    if name.endswith(".zip"):
        zf = zipfile.ZipFile(archive)
        entries = [(i.filename, i) for i in zf.infolist() if not i.is_dir()]
        read = zf.read
    else:
        zf = tarfile.open(archive, "r:*")
        entries = [(m.name, m) for m in zf.getmembers() if m.isfile()]
        read = lambda m: zf.extractfile(m).read()
    try:
        files = strip_wrapper([n for n, _ in entries])
        facts = {"glibc": set()}
        servers = [(f, e) for f, (_, e) in zip(files, entries)
                   if f.rsplit("/", 1)[-1] in (SERVER, SERVER + ".exe")]
        facts["program"], server = min(servers, key=lambda s: len(s[0])) if servers else (None, None)
        facts["kernel_gfx"] = sorted({g for f in files if f.split("/", 1)[0] in ("rocblas", "hipblaslt")
                                      or "gfx" in f.rsplit("/", 1)[-1] for g in GFX_RE.findall(f)})
        for f, (_, entry) in zip(files, entries):
            base = f.rsplit("/", 1)[-1]
            if HIP_DLL_RE.match(base) and "/" not in f:
                facts["hip_file_version"], facts["hip_description"] = pe_version(read(entry))
            elif entry is server or base.endswith(".so") or ".so." in base:
                data = read(entry)
                if data[:4] == b"\x7fELF":
                    facts["glibc"].update(GLIBC_RE.findall(data))
                    if entry is server:
                        facts.update(elf_notes(data))
    finally:
        zf.close()
    facts["glibc"] = max((tuple(int(n) for n in g.decode().split(".")) for g in facts["glibc"]),
                         default=None)
    return facts


def download(url, dest):
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def expected_digests(asset, assets_by_name):
    """Return the SHA-256 values GitHub and upstream's .sha256 sidecar publish for an asset."""
    digests = {}
    if (asset.get("digest") or "").startswith("sha256:"):
        digests["GitHub digest"] = asset["digest"].removeprefix("sha256:")
    sidecar = assets_by_name.get(asset["name"] + ".sha256")
    if sidecar:
        with urllib.request.urlopen(sidecar["browser_download_url"], timeout=60) as r:
            digests[sidecar["name"]] = r.read().decode("ascii").split()[0].lower()
    return digests


def describe(version, os_label, models, facts, upstream, rocm, hip):
    names = [m for m, _ in models]
    served = (", ".join(names[:-1]) + f", and {names[-1]}" if len(names) > 2 else
              " and ".join(names)) or "its supported GGUF models"
    description = (
        f"Gufo {version}, the open source inference engine built only for {DEVICE}, gfx1151, on "
        f"{os_label}. Serves {served} with OpenAI- and Anthropic-compatible APIs, images, tools, "
        "and speculative decoding (DFlash2, MTP, DSpark). Use it for these models on Strix Halo; "
        "use a general-purpose engine such as llama.cpp for other models or hardware."
    )

    model_parts = []
    for name, weights in models:
        hint = "; pass the first shard" if "shard" in weights.lower() else ""
        model_parts.append(f"{name} ({weights}{hint})")
    models_text = ("Only these GGUF models, from the exact files in the upstream model guides: "
                   + "; ".join(model_parts) + ". Other models are not supported." if model_parts else
                   "Only the GGUF models listed in the upstream docs/models guides.")
    if len(models_text) > 500:
        models_text = models_text[:497].rsplit(" ", 1)[0] + "..."

    notes = []
    if upstream["windows"]:
        notes.append(f"Windows {'11' if upstream['windows'] == '10.0.22000' else '10'} x64.")
    elif facts.get("glibc"):
        notes.append(f"Needs glibc {version_text(facts['glibc'])} or later.")
    gpus = "/".join(upstream["gpu"]) or "gfx1151"
    notes.append(f"Needs {DEVICE}, {gpus}, with a current AMD graphics driver.")
    if rocm:
        notes.append(f"Bundles the ROCm {rocm} (HIP {hip}) runtime that upstream validates on gfx1151."
                     if hip else f"Bundles the ROCm {rocm} runtime.")
    notes.append("Tuned for 128 GiB systems; smaller ones work for models that fit.")
    if upstream["os"] == "windows":
        notes.append("Upstream's production target is Linux; Windows is a supported native build.")
    compatibility = " ".join(notes)
    if len(compatibility) > 500:
        raise SystemExit(f"compatibility is over 500 characters: {compatibility}")
    return description, models_text, compatibility


def build(asset, release, ctx):
    match = ASSET_RE.match(asset["name"])
    version, os_name = match.group("version"), match.group("os")
    arch = ARCHES[match.group("arch")]
    stem = ARCHIVE_RE.sub("", asset["name"])
    name = re.sub(r"[._]", "-", stem.lower())
    if not NAME_RE.match(name) or len(name) > 64:
        raise SystemExit(f"{asset['name']}: '{name}' is not a valid bundle name")

    archive = Path(ctx["tmp"]) / asset["name"]
    try:
        sha256 = download(asset["browser_download_url"], archive)
        for source, expected in expected_digests(asset, ctx["assets"]).items():
            if expected != sha256:
                raise SystemExit(f"{asset['name']}: sha256 {sha256} does not match {source} {expected}")
        facts = inspect(archive, asset["name"])
    finally:
        archive.unlink(missing_ok=True)
    if facts["program"] is None:
        raise SystemExit(f"{asset['name']}: no {SERVER} executable in the archive")

    gpus = ctx["gpu"] or facts["kernel_gfx"]
    if ctx["gpu"] and facts["kernel_gfx"] and set(facts["kernel_gfx"]) - set(ctx["gpu"]):
        print(f"note: kernels also target {facts['kernel_gfx']}", file=sys.stderr)
    rocm = hip = None
    if facts.get("hip_file_version"):
        rocm = version_text(facts["hip_file_version"][:2])
    if facts.get("hip_description"):
        hip_match = re.search(r"amdhip64 ([0-9]+\.[0-9]+)", facts["hip_description"])
        hip = hip_match.group(1) if hip_match else None

    upstream = {"os": os_name, "gpu": gpus,
                "windows": ctx["windows"] if os_name == "windows" else None}
    description, models_text, compatibility = describe(
        version, OS_LABELS[os_name], ctx["models"], facts, upstream, rocm, hip)

    requires = {}
    if os_name == "windows" and ctx["windows"]:
        requires["os"] = {"windows": ctx["windows"]}
    elif os_name == "linux" and facts.get("kernel"):
        requires["os"] = {"linux": version_text(facts["kernel"])}
    if gpus:
        requires["gpu"] = gpus

    metadata = {
        "homepage": ctx["homepage"],
        "website": "https://gufo.blog",
        "release": release["html_url"],
        "asset": asset["name"],
        "revision": ctx["revision"],
        "os": os_name,
        "arch": arch,
        "backend": "rocm",
        "device": DEVICE,
    }
    if rocm:
        metadata["backend-version"] = rocm
    if hip:
        metadata["hip-version"] = hip
    if facts.get("glibc"):
        metadata["glibc"] = version_text(facts["glibc"])

    manifest = {
        "name": name,
        "description": description,
        "version": version,
        "platform": [f"{os_name}-{arch}-rocm"],
        "requires": requires,
        "download": {"url": asset["browser_download_url"], "sha256": sha256},
        "run": [f"bin/{facts['program']}", "serve", "llm", "--model", "{model}",
                "--host", "{host}", "--port", "{port}"],
        "api": API,
        "health": "/ready",
        "formats": ["gguf"],
        "models": models_text,
        "compatibility": compatibility,
        "license": "MIT",
        "metadata": metadata,
    }
    if not requires:
        del manifest["requires"]

    info = zipfile.ZipInfo("openengine.json", ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(Path(ctx["out"]) / f"{name}.oe", "w") as zf:
        zf.writestr(info, json.dumps(manifest, indent=2) + "\n")
    return f"made {name}.oe -> {asset['name']} (run {facts['program']})"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", default="gufo-org/gufo")
    parser.add_argument("--tag", help="release tag (default: newest release with engine assets)")
    parser.add_argument("--out", default="dist")
    args = parser.parse_args()

    release = find_release(args.repo, args.tag)
    tag = release["tag_name"]
    assets = [a for a in release["assets"] if ASSET_RE.match(a["name"])]
    if not assets:
        raise SystemExit(f"{args.repo} {tag} has no engine assets")
    print(f"{args.repo} {tag}: {len(assets)} engine assets", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ctx = {
            "out": out,
            "tmp": tmp,
            "assets": {a["name"]: a for a in release["assets"]},
            "homepage": f"https://github.com/{args.repo}",
            "revision": tag_revision(args.repo, tag),
            "gpu": hip_architectures(upstream_file(args.repo, "CMakeLists.txt", tag)),
            "models": gguf_models(upstream_file(args.repo, "docs/models/README.md", tag)),
            "windows": windows_baseline(upstream_file(args.repo, "tools/windows/check_bundle.py",
                                                      tag)),
        }
        if not ctx["models"]:
            print("warning: no GGUF models found in docs/models/README.md", file=sys.stderr)
        for asset in assets:
            print(build(asset, release, ctx), flush=True)

    made = sorted(p.name for p in out.glob("*.oe"))
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"tag={tag}\nurl={release['html_url']}\ncount={len(made)}\n")


if __name__ == "__main__":
    sys.exit(main())
