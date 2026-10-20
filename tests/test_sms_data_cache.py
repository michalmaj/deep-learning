"""Integrity and atomicity tests for the lab 09 SMS dataset cache."""

import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "labs" / "09_rnn" / "sms_data.py"


@pytest.fixture
def sms_data():
    spec = importlib.util.spec_from_file_location("lab09_sms_data", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive_bytes(payload: bytes = b"ham\thello\nspam\tfree prize\n") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("SMSSpamCollection", payload)
    return buffer.getvalue()


def _serve_archive(monkeypatch, sms_data, archive: bytes):
    monkeypatch.setattr(sms_data, "SMS_ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(sms_data, "urlopen", lambda *args, **kwargs: io.BytesIO(archive))


def test_download_and_extraction_are_committed_only_after_validation(
    tmp_path, monkeypatch, sms_data
):
    payload = b"ham\thello\nspam\tfree prize\n"
    archive = _archive_bytes(payload)
    _serve_archive(monkeypatch, sms_data, archive)

    dataset_path = sms_data.prepare_sms_spam_dataset(tmp_path)

    assert dataset_path.read_bytes() == payload
    assert (tmp_path / sms_data.SMS_ARCHIVE_NAME).read_bytes() == archive
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_valid_archive_repairs_corrupted_extracted_file_without_network(
    tmp_path, monkeypatch, sms_data
):
    payload = b"ham\thello\n"
    archive = _archive_bytes(payload)
    _serve_archive(monkeypatch, sms_data, archive)
    dataset_path = sms_data.prepare_sms_spam_dataset(tmp_path)
    dataset_path.write_bytes(b"corrupted")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("valid cached archive should avoid the network")

    monkeypatch.setattr(sms_data, "urlopen", fail_if_called)

    repaired_path = sms_data.prepare_sms_spam_dataset(tmp_path)

    assert repaired_path.read_bytes() == payload


def test_checksum_mismatch_raises_explicit_error_and_leaves_no_partial_cache(
    tmp_path, monkeypatch, sms_data
):
    archive = _archive_bytes()
    monkeypatch.setattr(sms_data, "SMS_ARCHIVE_SHA256", "0" * 64)
    monkeypatch.setattr(sms_data, "urlopen", lambda *args, **kwargs: io.BytesIO(archive))

    with pytest.raises(sms_data.DatasetIntegrityError, match="SHA-256 mismatch"):
        sms_data.prepare_sms_spam_dataset(tmp_path)

    assert not (tmp_path / sms_data.SMS_ARCHIVE_NAME).exists()
    assert not (tmp_path / sms_data.SMS_MEMBER_NAME).exists()
    assert list(tmp_path.iterdir()) == []


def test_interrupted_download_does_not_replace_existing_files(tmp_path, monkeypatch, sms_data):
    archive_path = tmp_path / sms_data.SMS_ARCHIVE_NAME
    dataset_path = tmp_path / sms_data.SMS_MEMBER_NAME
    tmp_path.mkdir(exist_ok=True)
    archive_path.write_bytes(b"previous archive")
    dataset_path.write_bytes(b"previous dataset")

    class InterruptedResponse(io.BytesIO):
        def __init__(self):
            super().__init__(b"partial")
            self.reads = 0

        def read(self, size=-1):
            self.reads += 1
            if self.reads > 1:
                raise OSError("connection lost")
            return super().read(size)

    monkeypatch.setattr(sms_data, "urlopen", lambda *args, **kwargs: InterruptedResponse())

    with pytest.raises(OSError, match="connection lost"):
        sms_data.prepare_sms_spam_dataset(tmp_path)

    assert archive_path.read_bytes() == b"previous archive"
    assert dataset_path.read_bytes() == b"previous dataset"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        sms_data.SMS_MEMBER_NAME,
        sms_data.SMS_ARCHIVE_NAME,
    ]


def test_invalid_cached_archive_is_replaced_by_verified_download(tmp_path, monkeypatch, sms_data):
    archive_path = tmp_path / sms_data.SMS_ARCHIVE_NAME
    tmp_path.mkdir(exist_ok=True)
    archive_path.write_bytes(b"truncated")
    archive = _archive_bytes()
    _serve_archive(monkeypatch, sms_data, archive)

    sms_data.prepare_sms_spam_dataset(tmp_path)

    assert archive_path.read_bytes() == archive


def test_failed_extraction_does_not_replace_existing_dataset(tmp_path, monkeypatch, sms_data):
    archive = _archive_bytes(b"ham\tnew data\n")
    _serve_archive(monkeypatch, sms_data, archive)
    dataset_path = sms_data.prepare_sms_spam_dataset(tmp_path)
    dataset_path.write_bytes(b"previous dataset")
    original_sha256_path = sms_data._sha256_path

    def fail_for_extraction_temporary(path):
        if path.suffix == ".tmp" and sms_data.SMS_MEMBER_NAME in path.name:
            raise OSError("disk read failed")
        return original_sha256_path(path)

    monkeypatch.setattr(sms_data, "_sha256_path", fail_for_extraction_temporary)

    with pytest.raises(OSError, match="disk read failed"):
        sms_data.prepare_sms_spam_dataset(tmp_path)

    assert dataset_path.read_bytes() == b"previous dataset"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        sms_data.SMS_MEMBER_NAME,
        sms_data.SMS_ARCHIVE_NAME,
    ]


def test_archive_must_contain_expected_dataset_member(tmp_path, monkeypatch, sms_data):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("unexpected.txt", b"data")
    archive = buffer.getvalue()
    _serve_archive(monkeypatch, sms_data, archive)

    with pytest.raises(sms_data.DatasetIntegrityError, match="does not contain"):
        sms_data.prepare_sms_spam_dataset(tmp_path)

    assert not (tmp_path / sms_data.SMS_MEMBER_NAME).exists()
