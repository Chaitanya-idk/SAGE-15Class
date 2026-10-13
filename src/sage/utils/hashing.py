import hashlib
import json
import os
from typing import Any, Dict, Union


def compute_file_hash(filepath: str, algorithm: str = "sha256", chunk_size: int = 65536) -> str:
    """Computes hexadecimal digest hash of a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found for hashing: {filepath}")

    hasher = getattr(hashlib, algorithm)()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_string_hash(text: str, algorithm: str = "sha256") -> str:
    """Computes hash of a string."""
    hasher = getattr(hashlib, algorithm)()
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def compute_dict_hash(d: Dict[str, Any], algorithm: str = "sha256") -> str:
    """Computes deterministic hash of a dictionary by sorting keys."""
    serialized = json.dumps(d, sort_keys=True, default=str)
    return compute_string_hash(serialized, algorithm=algorithm)


def compute_image_bytes_hash(data: bytes, algorithm: str = "md5") -> str:
    """Computes MD5 hash for image deduplication."""
    hasher = getattr(hashlib, algorithm)()
    hasher.update(data)
    return hasher.hexdigest()
