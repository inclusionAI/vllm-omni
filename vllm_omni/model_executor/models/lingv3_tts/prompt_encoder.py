# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LingV3-specific reference-audio encoder semantics."""

from typing import Any

import torch

from vllm_omni.model_executor.models.ming_tts.prompt_encoder import _resolve_prompt_latents


def resolve_prompt_latents(
    wrapper: Any,
    info: dict[str, Any],
    *,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor] | None:
    return _resolve_prompt_latents(
        wrapper,
        info,
        generator=generator,
        posterior_parameterization="softplus_scale",
    )
