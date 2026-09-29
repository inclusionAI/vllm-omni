# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared text-to-talker handoff for Ming speech pipelines."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def build_ming_talker_additional_info(
    text: str,
    prompt: OmniTokensPrompt | TextPrompt | Mapping | None,
) -> dict:
    """Preserve the same voice and generation defaults for sync/async handoffs."""
    if isinstance(prompt, Mapping):
        additional_info = prompt.get("additional_information") or prompt
    else:
        additional_info = getattr(prompt, "additional_information", None) or {}

    # Speaker embeddings may arrive as nested JSON lists.
    spk_emb = additional_info.get("spk_emb")
    if isinstance(spk_emb, list) and spk_emb and not hasattr(spk_emb[0], "device"):
        spk_emb = torch.tensor(spk_emb, dtype=torch.float32).unsqueeze(0)

    return {
        "ming_task": "omni",
        "text": text,
        "spk_emb": spk_emb,
        "voice_name": additional_info.get("voice_name", "DB30"),
        "prompt_text": additional_info.get("prompt_text"),
        "prompt_wav_lat": additional_info.get("prompt_wav_lat"),
        "prompt_wav_emb": additional_info.get("prompt_wav_emb"),
        "max_text_length": additional_info.get("max_text_length", 50),
    }


def build_ming_talker_inputs(
    source_outputs: list,
    prompt: OmniTokensPrompt | TextPrompt | list[OmniTokensPrompt | TextPrompt] | None = None,
) -> list[OmniTokensPrompt]:
    """Convert completed thinker text into self-contained Ming talker inputs."""
    prompts = prompt if isinstance(prompt, list) else [prompt]
    talker_inputs: list[OmniTokensPrompt] = []

    for index, source_output in enumerate(source_outputs):
        generated_text = getattr(source_output.outputs[0], "text", "") or ""
        original_prompt = prompts[index] if index < len(prompts) else None
        talker_inputs.append(
            OmniTokensPrompt(
                # The talker retokenizes text and builds its own embeddings.
                prompt_token_ids=[0],
                additional_information=build_ming_talker_additional_info(generated_text, original_prompt),
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


__all__ = ["build_ming_talker_additional_info", "build_ming_talker_inputs"]
