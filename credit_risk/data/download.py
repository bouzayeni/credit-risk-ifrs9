"""Download the Lending Club dataset from Kaggle and write a SHA-256 manifest.

Idempotent: if every source file listed in the manifest is present with a matching hash,
nothing is downloaded. Source files are never modified after extraction.

Usage (from the repo root):
    uv run python -m credit_risk.data.download
    uv run python -m credit_risk.data.download --from-zip path/to/archive.zip
    uv run python -m credit_risk.data.download --force
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from credit_risk.config import get_settings
from credit_risk.data.sources import KAGGLE_DOWNLOAD_URL, LICENSE, SOURCE_FILES

log = logging.getLogger("download")

MANIFEST_NAME = "_manifest.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS download URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "credit-risk-ifrs9"})
    with urllib.request.urlopen(request, timeout=300) as response, dest.open("wb") as fh:  # nosec B310
        shutil.copyfileobj(response, fh, length=1 << 20)


def extract_expected(zip_path: Path, raw_dir: Path) -> list[str]:
    """Extract only the known source files, flattened into raw_dir.

    Members are matched by basename and written to a fixed path, so archive entries such as
    "../x.csv.gz" or unexpected files can never be written outside raw_dir.
    """
    wanted = set(SOURCE_FILES)
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or name not in wanted or name in extracted:
                continue
            with zf.open(info) as src, (raw_dir / name).open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            extracted.append(name)
    return extracted


def build_manifest(raw_dir: Path, source: str) -> dict:
    files = {}
    for name in SOURCE_FILES:
        path = raw_dir / name
        if path.is_file():
            files[name] = {"sha256": sha256_of(path), "bytes": path.stat().st_size}
    return {
        "source": source,
        "license": LICENSE,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }


def manifest_is_current(raw_dir: Path) -> bool:
    manifest_path = raw_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in SOURCE_FILES:
        entry = manifest.get("files", {}).get(name)
        path = raw_dir / name
        if entry is None or not path.is_file() or sha256_of(path) != entry["sha256"]:
            return False
    return True


def run(raw_dir: Path, from_zip: Path | None = None, force: bool = False) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not force and manifest_is_current(raw_dir):
        log.info("Raw files match the manifest in %s; nothing to do.", raw_dir)
        return json.loads((raw_dir / MANIFEST_NAME).read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(dir=raw_dir) as tmp:
        if from_zip is not None:
            zip_path, source = from_zip, f"local archive {from_zip.name}"
        else:
            zip_path, source = Path(tmp) / "archive.zip", KAGGLE_DOWNLOAD_URL
            log.info("Downloading %s (about 650 MB)...", KAGGLE_DOWNLOAD_URL)
            download(KAGGLE_DOWNLOAD_URL, zip_path)
        extracted = extract_expected(zip_path, raw_dir)

    missing = sorted(set(SOURCE_FILES) - set(extracted))
    if missing:
        raise RuntimeError(f"Archive is missing expected files: {missing}")

    manifest = build_manifest(raw_dir, source)
    (raw_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Extracted %s into %s and wrote %s.", extracted, raw_dir, MANIFEST_NAME)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-zip", type=Path, help="Use a manually downloaded archive.")
    parser.add_argument("--force", action="store_true", help="Re-extract even if current.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(get_settings().raw_path, from_zip=args.from_zip, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
