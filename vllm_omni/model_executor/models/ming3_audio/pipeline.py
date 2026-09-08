# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-stage ASR topology for Ming-3.0-audio-flash."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

MING3_AUDIO_FLASH_PIPELINE = PipelineConfig(
    model_type="bailing_moe_v3_audio",
    default_deploy_config_name="ming3_audio_flash.yaml",
    model_arch="BailingMoeV3AudioForConditionalGeneration",
    hf_architectures=("BailingMoeV3AudioForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="asr",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="text_config",
            engine_output_type="text",
            model_arch="BailingMoeV3AudioForConditionalGeneration",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

__all__ = ["MING3_AUDIO_FLASH_PIPELINE"]
