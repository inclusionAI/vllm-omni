# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for Ming voice-preset registration."""

from pathlib import Path

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.model_executor.models.ming_flash_omni.voice_presets import VoicePresetRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_register_accepts_inference_mode_speaker_embedding(tmp_path: Path, mocker: MockerFixture) -> None:
    registry = VoicePresetRegistry(
        talker_dir=str(tmp_path),
        model_path=str(tmp_path),
        download_dir=None,
        audio_vae=None,
        aggregator=mocker.Mock(),
        spk_head=torch.nn.Linear(2, 4),
        patch_size=1,
    )
    with torch.inference_mode():
        raw_embedding = torch.ones(1, 2)
    mocker.patch.object(
        registry,
        "_load_single_wav",
        return_value=(torch.zeros(1, 4), raw_embedding),
    )

    registry.register("DB30", "prompt.wav", device=torch.device("cpu"), dtype=torch.float32)

    [projected] = registry.registered["DB30"]["spk_emb"]
    assert projected.shape == (1, 4)
    assert not projected.requires_grad
