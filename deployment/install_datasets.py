"""Install the verified Run 1 Zenodo archive without replacing existing files.

Run inside the deployment image; no private keys or model deserialization needed.
The public citation covers all versions, but deployment pins a reviewed release.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen


RECORD = "24122683"
ARCHIVE = "datasets_run1_complete.tar.zst"
URL = f"https://zenodo.org/records/{RECORD}/files/{ARCHIVE}?download=1"
SHA256 = "ffc3e6f3a469d0d1e189dc8d730d12cb73d1fe39926b89e85d8011a2cfd3d7e3"
ARCHIVE_BYTES = 10133888545
EXTRACTED_BYTES = 107368658653
MODEL_COUNT = 64
MARKER = ".run1-zenodo.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(cache):
    archive = cache / ARCHIVE
    partial = cache / (ARCHIVE + ".part")
    if not archive.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > ARCHIVE_BYTES:
            partial.unlink()
            offset = 0
        if offset < ARCHIVE_BYTES:
            headers = {"User-Agent": "BioAutoML-FAST-dataset-installer/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            print(f"Downloading {ARCHIVE} ({ARCHIVE_BYTES / 1024**3:.2f} GiB); resuming at {offset} bytes.", flush=True)
            with urlopen(Request(URL, headers=headers), timeout=120) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected download range; retry the download.")
                elif response.status == 200:
                    offset = 0  # Server does not support resuming: restart cleanly.
                else:
                    raise ValueError(f"Unexpected download status: {response.status}")
                written, last_report = offset, time.monotonic()
                with partial.open("ab" if offset else "wb") as stream:
                    while chunk := response.read(4 * 1024**2):
                        written += len(chunk)
                        if written > ARCHIVE_BYTES:
                            raise ValueError("Download exceeds expected archive size.")
                        stream.write(chunk)
                        if time.monotonic() - last_report >= 15:
                            print(f"Downloaded {written / ARCHIVE_BYTES:.1%}", flush=True)
                            last_report = time.monotonic()
        if partial.stat().st_size != ARCHIVE_BYTES:
            raise ValueError("Incomplete download; rerun the launcher and answer Yes to dataset setup to resume.")
        print("Verifying archive SHA-256...", flush=True)
        if digest(partial) != SHA256:
            partial.unlink()
            raise ValueError("Archive checksum mismatch; discarded the download. Existing datasets are unchanged.")
        partial.replace(archive)
    elif archive.stat().st_size != ARCHIVE_BYTES or digest(archive) != SHA256:
        raise ValueError(f"Cached archive failed verification. Remove {archive} and retry.")
    return archive


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    return Path(*path.parts)


def extract(archive, stage):
    print("Extracting Run 1 (approximately 100 GiB)...", flush=True)
    process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    total = 0
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as bundle:
            for member in bundle:
                safe_relative(member.name)
                if not (member.isdir() or member.isfile()):
                    raise ValueError(f"Unsupported archive member: {member.name}")
                total += member.size
                if total > EXTRACTED_BYTES:
                    raise ValueError("Archive exceeds expected extracted size.")
                if member.isdir():
                    # Keep staging directories writable even if source inputs were read-only.
                    (stage / safe_relative(member.name)).mkdir(parents=True, exist_ok=True)
                else:
                    bundle.extract(member, stage, filter="data")
        if process.wait() != 0:
            raise ValueError("Zstandard decompression failed.")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def validate_layout(root):
    models = list(root.glob("dataset*/runs/run_1/model/trained_model.sav"))
    if len(models) != MODEL_COUNT or not (root / "references.bib").is_file():
        raise ValueError("Run 1 archive is missing expected models or references.bib.")


def merge(stage, destination):
    files = sorted(p.relative_to(stage) for p in stage.rglob("*") if p.is_file())
    # Check every conflict before installing anything, including symlink parents.
    for relative in files:
        target = destination / relative
        for parent in [target, *target.parents]:
            if parent == destination:
                break
            if parent.is_symlink():
                raise ValueError(f"Existing symlink blocks installation: {parent}")
            if parent != target and parent.exists() and not parent.is_dir():
                raise ValueError(f"Existing file blocks directory: {parent}")
        if target.exists() and (not target.is_file() or digest(target) != digest(stage / relative)):
            raise ValueError(f"Existing file differs: {target}. Use a separate DATASETS_DIR; no files were replaced.")
    for relative in files:
        source, target = stage / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            # Same-filesystem link is atomic and never overwrites a destination.
            target.hardlink_to(source)
        source.unlink()
    return [str(p) for p in files]


def install(destination, cache):
    destination.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    with (destination / ".run1-install.lock").open("a") as lock, (cache / ".download.lock").open("a") as cache_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(cache_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = destination / MARKER
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved.get("sha256") != SHA256 or not saved.get("files"):
                raise ValueError("Existing installation marker belongs to a different release.")
            if not all((destination / safe_relative(p)).is_file() for p in saved["files"]):
                raise ValueError("Run 1 installation is incomplete. Use a fresh DATASETS_DIR to reinstall.")
            validate_layout(destination)
            print("Run 1 is already installed; preserving existing files, for signature verification.", flush=True)
            return
        margin = 1024**3
        needed_cache = max(0, ARCHIVE_BYTES - sum(p.stat().st_size for p in cache.glob(ARCHIVE + "*") if p.is_file()))
        needed_destination = EXTRACTED_BYTES + 3 * margin  # Includes space for the largest signing copy.
        if destination.stat().st_dev == cache.stat().st_dev:
            needed_destination += needed_cache
        if shutil.disk_usage(destination).free < needed_destination or shutil.disk_usage(cache).free < needed_cache + margin:
            raise ValueError("Insufficient disk space: allow about 113 GiB for archive, extraction, and overhead.")
        archive = download(cache)
        with tempfile.TemporaryDirectory(prefix=".run1-extract-", dir=destination) as temporary:
            stage = Path(temporary)
            extract(archive, stage)
            validate_layout(stage)
            files = merge(stage, destination)
        marker.write_text(json.dumps({"record": RECORD, "sha256": SHA256, "files": files}, indent=2))
        print(f"Installed Run 1 in {destination}. Archive retained in {cache}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    try:
        install(args.destination.resolve(), args.cache.resolve())
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"Dataset installation failed: {error}\n")
