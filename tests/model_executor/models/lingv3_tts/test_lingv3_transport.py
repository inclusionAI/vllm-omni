# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.lingv3_tts import (
    llm2audio_vae,
    llm2audio_vae_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


class _Request:
    external_req_id = "req-1"

    def __init__(self):
        self.finished = False

    def is_finished(self):
        return self.finished


def _manager():
    return SimpleNamespace(
        request_payload={},
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "initial_latent_chunk_size": 2,
                    "latent_chunk_size": 3,
                    "latent_left_context": 0,
                }
            }
        ),
    )


def _step(value: float):
    return {
        "lingv3_latent_patch": torch.full((1, 1, 64), value),
        "lingv3_has_patch": torch.tensor([True]),
    }


def test_async_transport_emits_initial_regular_and_final_chunks_once():
    manager = _manager()
    request = _Request()

    assert llm2audio_vae_async_chunk(manager, _step(1), request) is None
    initial = llm2audio_vae_async_chunk(manager, _step(2), request)
    assert initial.latent.shape == (2, 1, 64)
    assert not bool(initial.meta.finished)

    assert llm2audio_vae_async_chunk(manager, _step(3), request) is None
    assert llm2audio_vae_async_chunk(manager, _step(4), request) is None
    regular = llm2audio_vae_async_chunk(manager, _step(5), request)
    assert regular.latent.shape == (3, 1, 64)
    assert not bool(regular.meta.finished)

    request.finished = True
    final = llm2audio_vae_async_chunk(manager, _step(6), request)
    assert final.latent.shape == (1, 1, 64)
    assert bool(final.meta.finished)
    assert llm2audio_vae_async_chunk(manager, None, request) is None


def test_sync_transport_preserves_all_accumulated_latents():
    latents = torch.randn(7, 1, 64)
    stage_output = SimpleNamespace(
        finished=True,
        request_id="req-sync",
        outputs=[SimpleNamespace(multimodal_output={"lingv3_latent_patch": latents})],
    )
    prompts = llm2audio_vae([stage_output])
    assert len(prompts) == 1
    assert torch.equal(prompts[0]["additional_information"]["latent"], latents)
