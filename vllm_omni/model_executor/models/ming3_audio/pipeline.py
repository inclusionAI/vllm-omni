# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming-3.0-audio-flash speech-understanding and speech-output topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.ming3_audio"

_THINKER_STAGE = StagePipelineConfig(
    stage_id=0,
    model_stage="thinker",
    execution_type=StageExecutionType.LLM_AR,
    input_sources=(),
    final_output=True,
    final_output_type="text",
    owns_tokenizer=True,
    requires_multimodal_data=True,
    hf_config_name="text_config",
    engine_output_type="text",
    model_arch="BailingMoeV3AudioForConditionalGeneration",
    async_chunk_process_next_stage_input_func=f"{_PROC}.thinker2talker_async_chunk",
    sampling_constraints={"detokenize": True},
)

_TALKER_STAGE = StagePipelineConfig(
    stage_id=1,
    model_stage="ming_tts",
    model_arch="BailingTalker2",
    execution_type=StageExecutionType.LLM_GENERATION,
    input_sources=(0,),
    final_output=True,
    final_output_type="audio",
    engine_output_type="audio",
    tokenizer_subdir="talker/llm",
    sync_process_input_func=f"{_PROC}.thinker2talker_token_only",
)

MING3_AUDIO_FLASH_PIPELINE = PipelineConfig(
    model_type="bailing_moe_v3_audio",
    default_deploy_config_name="ming3_audio_flash.yaml",
    model_arch="BailingMoeV3AudioForConditionalGeneration",
    hf_architectures=("BailingMoeV3AudioForConditionalGeneration",),
    stages=(_THINKER_STAGE, _TALKER_STAGE),
)

MING3_AUDIO_FLASH_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="bailing_moe_v3_audio_thinker_only",
    default_deploy_config_name="ming3_audio_flash_thinker_only.yaml",
    model_arch="BailingMoeV3AudioForConditionalGeneration",
    hf_architectures=("BailingMoeV3AudioForConditionalGeneration",),
    stages=(_THINKER_STAGE,),
)

__all__ = [
    "MING3_AUDIO_FLASH_PIPELINE",
    "MING3_AUDIO_FLASH_THINKER_ONLY_PIPELINE",
]
