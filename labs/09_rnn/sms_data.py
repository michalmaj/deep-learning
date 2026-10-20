"""Safe local cache for the SMS Spam Collection used by lab 09."""

import hashlib
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

SMS_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SMS_ARCHIVE_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
SMS_ARCHIVE_NAME = "sms_spam_collection.zip"
SMS_MEMBER_NAME = "SMSSpamCollection"

_DOWNLOAD_TIMEOUT_SECONDS = 60
_MAX_ARCHIVE_BYTES = 5_000_000
_MAX_DATASET_BYTES = 2_000_000
_CHUNK_SIZE = 64 * 1024


class DatasetIntegrityError(RuntimeError):
    """The downloaded archive or extracted dataset failed validation."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_temporary_path(directory: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=".tmp")
    os.close(descriptor)
    return Path(name)


def _download_archive(destination: Path) -> None:
    temporary = _new_temporary_path(destination.parent, f".{destination.name}.")
    digest = hashlib.sha256()
    total_bytes = 0

    try:
        with urlopen(SMS_ARCHIVE_URL, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            with temporary.open("wb") as file:
                while chunk := response.read(_CHUNK_SIZE):
                    total_bytes += len(chunk)
                    if total_bytes > _MAX_ARCHIVE_BYTES:
                        raise DatasetIntegrityError(
                            f"SMS archive exceeds {_MAX_ARCHIVE_BYTES} bytes"
                        )
                    digest.update(chunk)
                    file.write(chunk)

        actual_sha256 = digest.hexdigest()
        if actual_sha256 != SMS_ARCHIVE_SHA256:
            raise DatasetIntegrityError(
                f"SMS archive SHA-256 mismatch: expected {SMS_ARCHIVE_SHA256}, got {actual_sha256}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_archive(destination: Path) -> None:
    if destination.is_file() and _sha256_path(destination) == SMS_ARCHIVE_SHA256:
        return
    _download_archive(destination)


def _sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _extract_dataset(archive_path: Path, destination: Path) -> None:
    temporary: Path | None = None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            try:
                member = archive.getinfo(SMS_MEMBER_NAME)
            except KeyError as error:
                raise DatasetIntegrityError(
                    f"SMS archive does not contain {SMS_MEMBER_NAME}"
                ) from error

            if member.is_dir() or member.file_size <= 0:
                raise DatasetIntegrityError("SMS dataset member must be a non-empty file")
            if member.file_size > _MAX_DATASET_BYTES:
                raise DatasetIntegrityError(
                    f"SMS dataset exceeds {_MAX_DATASET_BYTES} extracted bytes"
                )

            with archive.open(member) as source:
                expected_sha256 = _sha256_stream(source)
            if (
                destination.is_file()
                and destination.stat().st_size == member.file_size
                and _sha256_path(destination) == expected_sha256
            ):
                return

            temporary = _new_temporary_path(destination.parent, f".{destination.name}.")
            with archive.open(member) as source, temporary.open("wb") as target:
                copied_bytes = 0
                while chunk := source.read(_CHUNK_SIZE):
                    copied_bytes += len(chunk)
                    if copied_bytes > _MAX_DATASET_BYTES:
                        raise DatasetIntegrityError(
                            f"SMS dataset exceeds {_MAX_DATASET_BYTES} extracted bytes"
                        )
                    target.write(chunk)

            if copied_bytes != member.file_size or _sha256_path(temporary) != expected_sha256:
                raise DatasetIntegrityError("extracted SMS dataset failed integrity validation")
            os.replace(temporary, destination)
    except zipfile.BadZipFile as error:
        raise DatasetIntegrityError("SMS archive is not a valid ZIP file") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_sms_spam_dataset(cache_dir: str | Path) -> Path:
    """Return a verified dataset path, downloading and extracting atomically."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / SMS_ARCHIVE_NAME
    dataset_path = cache_dir / SMS_MEMBER_NAME
    _ensure_archive(archive_path)
    _extract_dataset(archive_path, dataset_path)
    return dataset_path
