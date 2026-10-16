# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Normalize a Ming TTS deploy/talker checkpoint for the engine."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def prepare_talker_checkpoint(source: str, cache_dir: str | None = None) -> str:
    """Link deployment assets into an engine config view; leave sources intact."""
    deployment = Path(source).resolve()
    # The engine may call the resolver again on our already normalized view.
    if (deployment / "config.json").is_file():
        config = json.loads((deployment / "config.json").read_text())
        if config.get("model_type") == "ming_flash_omni_talker":
            return str(deployment)
    root = (deployment / "talker").resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"Expected deploy/talker/config.json at {deployment}")
    with (root / "config.json").open() as handle:
        config = json.load(handle)
    if "BailingTalker2" not in config.get("architectures", []):
        raise ValueError(f"Expected a converted BailingTalker2 checkpoint at {root}")
    for required in ("llm/config.json", "vae/config.json", "vae/model.safetensors", "campplus.onnx"):
        if not (root / required).is_file():
            raise FileNotFoundError(f"Incomplete talker checkpoint: {root / required}")
    if not list(root.glob("model*.safetensors")):
        raise FileNotFoundError(f"No talker safetensors in {root}")
    with (root / "llm/config.json").open() as handle:
        config["llm_config"] = json.load(handle)
    config.update(
        model_type="ming_flash_omni_talker",
        strict_weight_loading=True,
        max_decode_steps=1000,
        max_cache_len=2304,
        # Keep AntOC text handling; use native cached VAE decoding by default.
        normalize_text=False,
        duration_limit=False,
        stream_decode=True,
        # None follows native Ming: trim full output, retain streaming chunks.
        trim_trailing_silence=None,
    )
    config_json = json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    # A new view must not reuse an older cache entry that linked deploy/data.
    digest = hashlib.sha256(("no-preset-data-v1\0" + str(deployment) + config_json).encode()).hexdigest()[:20]
    cache = Path(cache_dir or os.environ.get("MING_TTS_CONFIG_CACHE", "~/.cache/vllm-omni/ming_talker")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / digest
    if target.is_dir():
        if (target / "config.json").read_text() != config_json:
            raise RuntimeError(f"Conflicting talker config cache at {target}")
        return str(target)
    temporary = Path(tempfile.mkdtemp(prefix=f".{digest}-", dir=cache))
    try:
        for entry in root.iterdir():
            if entry.name not in {"config.json", "data"}:
                (temporary / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        (temporary / "config.json").write_text(config_json)
        try:
            temporary.rename(target)
        except OSError:
            if not target.is_dir() or (target / "config.json").read_text() != config_json:
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(target)


def resolve_ming_talker_checkpoint(model: str, revision: str | None, task_type: str | None) -> str:
    """Resolve an OSS deploy/ URI or a local copy of that exact layout."""
    del revision, task_type
    source = os.environ.get("MING_TTS_MODEL_PATH") or model
    if source.startswith("oss://"):
        from .oss_checkpoint import download_oss_talker

        source = download_oss_talker(source)
    elif not Path(source).is_dir():
        raise ValueError("Ming TTS expects an oss:// deploy/ URI or a local deploy/ directory")
    return prepare_talker_checkpoint(source)
