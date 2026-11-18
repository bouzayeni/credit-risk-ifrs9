import json
import zipfile
from pathlib import Path

import pytest

from credit_risk.data import download
from credit_risk.data.sources import ACCEPTED_FILE


def make_archive(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_extracts_only_expected_files_and_writes_manifest(tmp_path):
    archive = make_archive(
        tmp_path / "a.zip",
        {
            f"accepted/{ACCEPTED_FILE}": b"loan data",
            "rejected_2007_to_2018Q4.csv.gz": b"not used",
            f"../{ACCEPTED_FILE}x": b"traversal attempt",
        },
    )
    raw = tmp_path / "raw"
    manifest = download.run(raw, from_zip=archive)

    assert sorted(p.name for p in raw.iterdir()) == [download.MANIFEST_NAME, ACCEPTED_FILE]
    assert (raw / ACCEPTED_FILE).read_bytes() == b"loan data"
    assert manifest["files"][ACCEPTED_FILE]["sha256"] == download.sha256_of(raw / ACCEPTED_FILE)
    assert json.loads((raw / download.MANIFEST_NAME).read_text())["license"].startswith("CC0")


def test_is_idempotent_and_detects_changed_files(tmp_path):
    archive = make_archive(tmp_path / "a.zip", {ACCEPTED_FILE: b"v1"})
    raw = tmp_path / "raw"
    download.run(raw, from_zip=archive)
    assert download.manifest_is_current(raw)

    (raw / ACCEPTED_FILE).write_bytes(b"tampered")
    assert not download.manifest_is_current(raw)


def test_missing_file_in_archive_raises(tmp_path):
    archive = make_archive(tmp_path / "a.zip", {"other.csv": b"x"})
    with pytest.raises(RuntimeError, match="missing expected files"):
        download.run(tmp_path / "raw", from_zip=archive)


def test_refuses_non_https(tmp_path):
    with pytest.raises(ValueError):
        download.download("http://example.com/x.zip", tmp_path / "x.zip")
