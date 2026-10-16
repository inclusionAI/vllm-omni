# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming3 chat voice controls survive the thinker-to-talker handoff."""

import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.model_executor.stage_input_processors.ming_talker import build_ming_talker_additional_info

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _serving_chat() -> OmniOpenAIServingChat:
    chat = object.__new__(OmniOpenAIServingChat)
    chat.engine_client = SimpleNamespace(
        stage_configs=[
            {"engine_args": {"model_arch": "BailingMoeV3AudioForConditionalGeneration", "model_stage": "thinker"}},
            {"engine_args": {"model_arch": "BailingTalker2", "model_stage": "ming_tts"}},
        ]
    )
    chat.model_config = SimpleNamespace(allowed_local_media_path="", allowed_media_domains=None)
    return chat


def test_chat_ip_name_passes_to_talker_without_reference():
    info = build_ming_talker_additional_info(
        "你好", {"additional_information": {"voice_mode": "ip", "ip_name": "灵小柔"}}
    )

    assert json.loads(info["instruction"])["audio_sequence"][0]["IP"] == "灵小柔"
    assert info["prompt_audio_mode"] == "none"
    assert info["reference_audio"] is None


@pytest.mark.asyncio
async def test_chat_base64_reference_passes_decoded_samples_to_talker(mocker: MockerFixture):
    chat = _serving_chat()
    encoded = base64.b64encode(b"RIFFfakeWAVE").decode()
    prompt = {"additional_information": {"ref_audio": encoded}}
    connector = mocker.patch("vllm_omni.entrypoints.openai.serving_chat.MediaConnector").return_value
    connector.fetch_audio_async = mocker.AsyncMock(return_value=(np.array([0.1, -0.1], dtype=np.float32), 16000))

    await chat._attach_ming3_reference_audio(prompt)
    info = build_ming_talker_additional_info("你好", prompt)

    connector.fetch_audio_async.assert_awaited_once_with(f"data:audio/wav;base64,{encoded}")
    assert "ref_audio" not in prompt["additional_information"]
    assert info["speaker_conditioning"] == "prompt"
    assert info["prompt_audio_mode"] == "speaker_only"
    assert info["reference_audio"][0]["sample_rate"] == 16000
    assert info["reference_audio"][0]["samples"] == pytest.approx([0.1, -0.1])


@pytest.mark.asyncio
async def test_chat_rejects_ip_and_reference_together():
    chat = _serving_chat()
    prompt = {"additional_information": {"voice_mode": "ip", "ip_name": "灵小柔", "ref_audio": "AA=="}}

    with pytest.raises(ValueError, match="either an IP voice or ref_audio"):
        await chat._attach_ming3_reference_audio(prompt)
