# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE, AudioVAEConfig
from vllm_omni.model_executor.models.lingv3_tts.lingv3_tts_audio_vae import LingV3TTSAudioVAEModel
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import KEY_REQUEST_ID
from vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae import MingAudioVAEModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def _stage(stage_type, patch_size):
    torch.manual_seed(17)
    flow_patch_size = 4 if patch_size == -1 else 1
    config = AudioVAEConfig(
        patch_size=patch_size,
        dec_kwargs={
            "backbone": {
                "vocab_size": 1,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "max_position_embeddings": 256,
                "_attn_implementation": "eager",
            },
            "latent_dim": 2,
            "output_dim": 4,
        },
    )
    stage = stage_type.__new__(stage_type)
    torch.nn.Module.__init__(stage)
    stage.audio = AudioVAE(config, build_encoder=False).eval()
    stage.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu")))
    stage.ming_config = SimpleNamespace(sample_rate=100, latent_dim=2, patch_size=flow_patch_size)
    stage._past_key_values, stage._stream_state = {}, {}
    stage._patch_totals, stage._sample_totals = {}, {}
    stage._state_access_times = OrderedDict()
    return stage


def _forward(stage, latent, finished, request_id="request"):
    output = stage.forward(
        model_intermediate_buffer=[
            {KEY_REQUEST_ID: request_id, "latent": latent, "meta": {"stream_finished": finished}}
        ]
    )
    return output.multimodal_outputs["model_outputs"][0]


@pytest.mark.parametrize("stage_type", [MingAudioVAEModel, LingV3TTSAudioVAEModel])
@pytest.mark.parametrize("patch_size", [4, -1])
@pytest.mark.parametrize("patches", [1, 2, 4, 10, 27])
@pytest.mark.parametrize("terminal", ["none", "empty"])
@torch.inference_mode()
def test_empty_eos_flushes_the_same_audio_as_final_latent(stage_type, patch_size, patches, terminal):
    stage = _stage(stage_type, patch_size)
    terminal = None if terminal == "none" else torch.empty(0, stage.ming_config.patch_size, 2)
    latent = torch.randn(patches, stage.ming_config.patch_size, 2)
    expected = _forward(stage, latent, True)
    prefix = _forward(stage, latent, False)
    tail = _forward(stage, terminal, True)

    assert tail.numel() > 0
    torch.testing.assert_close(torch.cat([prefix, tail]), expected)
    assert _forward(stage, terminal, True).numel() == 0
    assert not any(
        "request" in state
        for state in (stage._past_key_values, stage._stream_state, stage._patch_totals, stage._sample_totals)
    )
    assert "request" not in stage._state_access_times


@torch.inference_mode()
def test_empty_eos_preserves_other_request_state():
    stage = _stage(LingV3TTSAudioVAEModel, 4)
    latent_a, latent_b = torch.randn(4, 1, 2), torch.randn(10, 1, 2)
    expected_a = _forward(stage, latent_a, True, "a")
    expected_b = _forward(stage, latent_b, True, "b")
    prefix_a = _forward(stage, latent_a, False, "a")
    prefix_b = _forward(stage, latent_b, False, "b")
    tail_a = _forward(stage, None, True, "a")
    assert "b" in stage._stream_state
    tail_b = _forward(stage, torch.empty(0, 1, 2), True, "b")

    torch.testing.assert_close(torch.cat([prefix_a, tail_a]), expected_a)
    torch.testing.assert_close(torch.cat([prefix_b, tail_b]), expected_b)
    assert not stage._stream_state and not stage._past_key_values
