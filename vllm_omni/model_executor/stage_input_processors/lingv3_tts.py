# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage transport for LingV3 continuous latents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.lingv3_tts.constants import (
    INITIAL_LATENT_CHUNK_SIZE,
    LATENT_CHUNK_SIZE,
    LATENT_DIM,
)
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_CHUNK_ID as STATE_CHUNK_ID,
)
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_REQUEST_ID as STATE_REQUEST_ID,
)


def _patches(output: Mapping[str, Any] | None) -> torch.Tensor | None:
    if not isinstance(output, Mapping):
        return None
    patch = output.get("lingv3_latent_patch")
    active = output.get("lingv3_has_patch")
    if not isinstance(patch, torch.Tensor) or patch.numel() == 0:
        return None
    if patch.ndim == 2:
        patch = patch.unsqueeze(0)
    if patch.ndim != 3 or patch.shape[-2:] != (1, LATENT_DIM):
        raise ValueError(f"Invalid LingV3 latent tensor shape: {tuple(patch.shape)}")
    if isinstance(active, torch.Tensor) and active.numel():
        indices = (active.reshape(-1) > 0).nonzero(as_tuple=True)[0]
        if not indices.numel():
            return None
        patch = patch.index_select(0, indices.to(patch.device))
    return patch.detach().to("cpu", dtype=torch.float32)


def _chunk_size(transfer_manager: Any, chunk_index: int) -> int:
    connector = getattr(transfer_manager, "connector", None)
    raw = getattr(connector, "config", {}) or {}
    config = raw.get("extra", raw) if isinstance(raw, Mapping) else {}
    chunk = int(config.get("latent_chunk_size", LATENT_CHUNK_SIZE))
    initial = int(config.get("initial_latent_chunk_size", INITIAL_LATENT_CHUNK_SIZE))
    left = int(config.get("latent_left_context", 0))
    if chunk <= 0 or initial < 0 or initial > chunk:
        raise ValueError(f"Invalid LingV3 chunk sizes: initial={initial}, chunk={chunk}")
    if left != 0:
        raise ValueError("LingV3 streaming AudioVAE uses cached state; latent_left_context must be 0")
    schedule = config.get("latent_chunk_size_schedule")
    if schedule is not None:
        if not isinstance(schedule, (list, tuple)) or not schedule:
            raise ValueError("LingV3 latent_chunk_size_schedule must be a non-empty list of positive integers")
        if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in schedule):
            raise ValueError("LingV3 latent_chunk_size_schedule must contain only positive integers")
        return schedule[min(chunk_index, len(schedule) - 1)]
    return initial if chunk_index == 0 and initial else chunk


def llm2audio_vae_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    # TODO: Share the Ming/Ling chunk-buffer state machine after streaming parity.
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    state_root = transfer_manager.request_payload.get(request_id)
    if not isinstance(state_root, dict) or "_lingv3_async_state" not in state_root:
        state_root = {"_lingv3_async_state": {"seen": 0, "chunks": 0, "terminal": False}}
        transfer_manager.request_payload[request_id] = state_root
    state = state_root["_lingv3_async_state"]
    if state["terminal"]:
        return None

    current = _patches(multimodal_output)
    if current is not None:
        transfer_manager.code_prompt_token_ids[request_id].extend(list(current))
    buffered = transfer_manager.code_prompt_token_ids[request_id]
    seen = int(state["seen"])
    available = len(buffered) - seen
    target = _chunk_size(transfer_manager, int(state["chunks"]))
    if available <= 0:
        if not finished:
            return None
        state["terminal"] = True
        return OmniPayloadStruct(
            codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
            meta=MetaStruct(
                finished=torch.tensor(True),
                stream_finished=torch.tensor(True),
            ),
            kv_metadata={
                STATE_REQUEST_ID: request_id,
                STATE_CHUNK_ID: int(transfer_manager.put_req_chunk[request_id]),
            },
        )

    if not finished and available < target:
        return None
    count = available if finished else target
    latent = torch.stack(buffered[seen : seen + count], dim=0)
    state["seen"] = seen + count
    state["chunks"] += 1
    state["terminal"] = finished
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor([0], dtype=torch.long)),
        meta=MetaStruct(
            finished=torch.tensor(finished),
            stream_finished=torch.tensor(finished),
        ),
        latent=latent,
        kv_metadata={
            STATE_REQUEST_ID: request_id,
            STATE_CHUNK_ID: int(transfer_manager.put_req_chunk[request_id]),
            "lingv3_emit_patch_count": count,
        },
    )


def llm2audio_vae(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any = None,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data, streaming_context
    outputs: list[OmniTokensPrompt] = []
    for stage_output in source_outputs:
        if not bool(getattr(stage_output, "finished", True)):
            continue
        output = stage_output.outputs[0]
        latent = _patches(output.multimodal_output)
        outputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information={
                    "latent": latent if latent is not None else torch.zeros((0, 1, LATENT_DIM), dtype=torch.float32),
                    STATE_REQUEST_ID: getattr(stage_output, "request_id", None),
                    "finished": torch.tensor(True),
                },
            )
        )
    return outputs


__all__ = ["llm2audio_vae", "llm2audio_vae_async_chunk"]
