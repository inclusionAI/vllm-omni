# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cache the fixed OSS deploy/talker inference layout."""

from __future__ import annotations

import configparser
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


def _open_bucket(name: str):
    try:
        import oss2
    except ImportError as exc:
        raise RuntimeError("Reading oss:// talker checkpoints requires oss2; install it with pip install oss2") from exc

    config = configparser.ConfigParser(interpolation=None)
    explicit = os.getenv("MING_TTS_OSS_CONFIG") or os.getenv("OSSUTIL_CONFIG_FILE")
    candidates = (
        [Path(explicit).expanduser()]
        if explicit
        else [
            Path.home() / ".ossutilconfig",
            *(parent / f".ossutilconfig_{name}" for parent in (Path.cwd(), *Path.cwd().parents)),
        ]
    )
    for path in candidates:
        if path.is_file():
            config.read(path)
            break
    credentials = config["Credentials"] if config.has_section("Credentials") else {}
    endpoint = os.getenv("OSS_ENDPOINT") or credentials.get("endpoint")
    key = os.getenv("OSS_ACCESS_KEY_ID") or credentials.get("accesskeyid")
    secret = os.getenv("OSS_ACCESS_KEY_SECRET") or credentials.get("accesskeysecret")
    token = os.getenv("OSS_SESSION_TOKEN") or credentials.get("sts_token")
    if not endpoint or not key or not secret:
        raise RuntimeError(
            "Configure OSS_ENDPOINT/OSS_ACCESS_KEY_ID/OSS_ACCESS_KEY_SECRET, or set "
            "MING_TTS_OSS_CONFIG to an ossutil config with a [Credentials] section"
        )
    if not endpoint.startswith(("http://", "https://")):
        endpoint = "https://" + endpoint
    auth = oss2.StsAuth(key, secret, token) if token else oss2.Auth(key, secret)
    return oss2.Bucket(auth, endpoint, name)


def _list_objects(bucket, prefix):
    import oss2

    return oss2.ObjectIterator(bucket, prefix=prefix)


def _is_inference_file(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/") or ".." in parts or "\\" in relative:
        raise ValueError(f"Invalid checkpoint object path: {relative!r}")
    if len(parts) == 1:
        return relative in {"config.json", "campplus.onnx"} or (
            relative.startswith("model") and relative.endswith((".safetensors", ".safetensors.index.json"))
        )
    return (
        len(parts) == 2
        and parts[0] in {"llm", "vae"}
        and parts[1].endswith((".json", ".safetensors", ".txt", ".jinja"))
    )


def download_oss_talker(uri: str, cache_dir: str | None = None) -> str:
    """Cache deploy/talker weights without loading preset voice data.

    ETags version the cache; OSS SDK verifies transfer CRCs. Completed files
    and metadata are published together under a process-shared lock.
    """
    parsed = urlsplit(uri)
    if parsed.scheme != "oss" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Expected oss://bucket/checkpoint-prefix/ without query or fragment")
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    if prefix == "/":
        raise ValueError("An OSS checkpoint prefix is required")
    bucket = _open_bucket(parsed.netloc)
    if not bucket.object_exists(prefix + "talker/config.json"):
        raise FileNotFoundError(f"Expected talker/config.json under {uri}")
    objects = []
    for obj in _list_objects(bucket, prefix + "talker/"):
        if not obj.key.endswith("/") and _is_inference_file(obj.key[len(prefix + "talker/") :]):
            objects.append({"name": obj.key[len(prefix) :], "size": obj.size, "etag": obj.etag})
    names = {item["name"] for item in objects}
    required = {
        "talker/config.json",
        "talker/campplus.onnx",
        "talker/llm/config.json",
        "talker/vae/config.json",
        "talker/vae/model.safetensors",
    }
    if required - names or not any(name.startswith("talker/model") and name.endswith(".safetensors") for name in names):
        raise FileNotFoundError(f"Incomplete deploy/talker export at {uri}")
    objects.sort(key=lambda item: item["name"])
    snapshot = {"bucket": parsed.netloc, "prefix": prefix, "objects": objects}
    encoded = json.dumps(snapshot, sort_keys=True)
    digest = hashlib.sha256(encoded.encode()).hexdigest()[:24]
    cache = Path(cache_dir or os.getenv("MING_TTS_OSS_CACHE", "~/.cache/vllm-omni/oss_talker")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / digest
    with (cache / f"{digest}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = target / "_oss_snapshot.json"
        if (
            marker.is_file()
            and marker.read_text() == encoded
            and all(
                (target / item["name"]).is_file() and (target / item["name"]).stat().st_size == item["size"]
                for item in objects
            )
        ):
            return str(target)
        temporary = Path(tempfile.mkdtemp(prefix=f".{digest}-", dir=cache))
        try:

            def download(item):
                name = item["name"]
                path = temporary / name
                path.parent.mkdir(parents=True, exist_ok=True)
                bucket.get_object_to_file(prefix + name, str(path))
                if path.stat().st_size != item["size"]:
                    raise RuntimeError(f"Incomplete OSS download: {name}")

            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(download, objects))
            (temporary / "_oss_snapshot.json").write_text(encoded)
            if target.exists():
                shutil.rmtree(target)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return str(target)
