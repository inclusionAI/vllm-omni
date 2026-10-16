# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming talker checkpoint resolution without preset voice data."""

import json
from pathlib import Path

import pytest

from vllm_omni.model_executor.models.ming_flash_omni.checkpoint import prepare_talker_checkpoint

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_prepare_talker_checkpoint_does_not_require_or_link_data(tmp_path: Path):
    root = tmp_path / "deployment"
    talker = root / "talker"
    (talker / "llm").mkdir(parents=True)
    (talker / "vae").mkdir()
    (root / "data").mkdir()
    (talker / "config.json").write_text(json.dumps({"architectures": ["BailingTalker2"]}))
    (talker / "llm/config.json").write_text("{}")
    (talker / "vae/config.json").write_text("{}")
    for name in ("model.safetensors", "vae/model.safetensors", "campplus.onnx"):
        (talker / name).touch()
    (root / "data/voice_name.json").write_text("{}")

    view = Path(prepare_talker_checkpoint(str(root), cache_dir=str(tmp_path / "cache")))

    assert not (view / "data").exists()
    assert (view / "model.safetensors").resolve() == talker / "model.safetensors"

    (root / "data/voice_name.json").unlink()
    assert prepare_talker_checkpoint(str(root), cache_dir=str(tmp_path / "cache")) == str(view)
