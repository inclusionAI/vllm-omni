# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processors for Ming-3.0-audio-flash."""

from __future__ import annotations

from typing import Any

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

from .ming_talker import build_ming_talker_inputs


def thinker2talker_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | list[Any] | None = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build Ming talker inputs after the thinker has completed its answer."""
    return build_ming_talker_inputs(source_outputs, prompt)


__all__ = ["thinker2talker_token_only"]
