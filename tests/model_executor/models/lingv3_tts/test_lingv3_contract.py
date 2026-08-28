# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.models.common.ming.continuous_tts import (
    STOP_REASON_CONTINUE,
    STOP_REASON_MAX_DECODE_STEPS,
    STOP_REASON_STOP_HEAD,
    resolve_stop_decision,
)
from vllm_omni.model_executor.models.lingv3_tts.config_lingv3_tts import (
    LingV3TTSConfig,
    LingV3TTSRuntimeConfig,
)
from vllm_omni.model_executor.models.lingv3_tts.constants import (
    AUDIO_END_TOKEN_ID,
    AUDIO_PATCH_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    SPEAKER_END_TOKEN_ID,
    SPEAKER_START_TOKEN_ID,
)
from vllm_omni.model_executor.models.lingv3_tts.prompt_assembly import (
    ASSISTANT_PREFIX,
    TEXT_PREFIX,
    USER_PREFIX,
    build_lingv3_prompt,
    build_lingv3_prompt_token_ids,
    validate_lingv3_tokenizer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


class FakeTokenizer:
    eos_token_id = 156892
    special = {
        "<audioPatch>": AUDIO_PATCH_TOKEN_ID,
        "<audio>": AUDIO_START_TOKEN_ID,
        "</audio>": AUDIO_END_TOKEN_ID,
        "<spk>": SPEAKER_START_TOKEN_ID,
        "</spk>": SPEAKER_END_TOKEN_ID,
    }

    def convert_tokens_to_ids(self, token):
        return self.special.get(token)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self.special:
            return [self.special[text]]
        return [1000 + ord(char) for char in text]


def _config() -> LingV3TTSConfig:
    backbone = {
        "model_type": "bailing_moe_v3",
        "vocab_size": 157184,
        "hidden_size": 1536,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "layer_group_size": 4,
        "head_dim": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "short_conv_kernel_size": 4,
    }
    audio = {
        "sample_rate": 44100,
        "patch_size": 4,
        "semantic_module_kwargs": None,
        "enc_kwargs": {"hop_size": 882, "latent_dim": 64},
        "dec_kwargs": {"output_dim": 882, "latent_dim": 64},
    }
    return LingV3TTSConfig(
        llm_config=backbone,
        audio_config=audio,
        tts_config={
            "sample_rate": 44100,
            "latent_dim": 64,
            "speaker_embedding_dim": 192,
            "patch_size": 1,
            "history_size": 1,
            "audio_patch_token_id": AUDIO_PATCH_TOKEN_ID,
            "audio_start_token_id": AUDIO_START_TOKEN_ID,
            "audio_end_token_id": AUDIO_END_TOKEN_ID,
            "speaker_start_token_id": SPEAKER_START_TOKEN_ID,
            "speaker_end_token_id": SPEAKER_END_TOKEN_ID,
        },
        aggregator_config={"in_channels": 64, "hidden_size": 1024},
        flow_config={"z_channels": 64, "hidden_size": 1024},
    )


def test_config_normalizes_bailing_and_keeps_independent_geometry():
    config = _config()
    runtime = LingV3TTSRuntimeConfig.from_hf_config(config)
    runtime.validate()
    assert config.llm_config.model_type == "bailing_hybrid"
    assert config.llm_config.checkpoint_model_type == "bailing_moe_v3"
    assert config.llm_config.kv_lora_rank + config.llm_config.qk_rope_head_dim == 576
    assert runtime.patch_size == runtime.history_patch_size == 1
    assert runtime.vae_patch_size == 4


def test_prompt_contract_speaker_only():
    tokenizer = FakeTokenizer()
    ids, speaker_position, prompt_position = build_lingv3_prompt_token_ids(tokenizer, target_text="target")
    expected = (
        tokenizer.encode(USER_PREFIX)
        + [SPEAKER_START_TOKEN_ID, AUDIO_PATCH_TOKEN_ID, SPEAKER_END_TOKEN_ID]
        + tokenizer.encode(TEXT_PREFIX)
        + tokenizer.encode("target")
        + tokenizer.encode(ASSISTANT_PREFIX)
        + [AUDIO_START_TOKEN_ID]
    )
    assert ids == expected
    assert ids[speaker_position] == AUDIO_PATCH_TOKEN_ID
    assert prompt_position is None


def test_prompt_contract_requires_reference_audio():
    with pytest.raises(ValueError, match="requires reference audio"):
        build_lingv3_prompt(
            FakeTokenizer(),
            target_text="target",
            reference_waveform=None,
            reference_sample_rate=24000,
        )


def test_prompt_contract_concatenates_reference_and_target_without_separator():
    tokenizer = FakeTokenizer()
    ids, _, prompt_position = build_lingv3_prompt_token_ids(
        tokenizer,
        target_text="target",
        prompt_text="reference",
        prompt_latent_steps=3,
    )
    joined = tokenizer.encode("reference") + tokenizer.encode("target")
    text_start = len(tokenizer.encode(USER_PREFIX)) + 3 + len(tokenizer.encode(TEXT_PREFIX))
    assert ids[text_start : text_start + len(joined)] == joined
    assert ids[prompt_position:] == [AUDIO_PATCH_TOKEN_ID] * 3


def test_tokenizer_mismatch_is_not_repaired():
    tokenizer = FakeTokenizer()
    tokenizer.special["<audio>"] = -1
    with pytest.raises(ValueError, match="expected 157165"):
        validate_lingv3_tokenizer(tokenizer)


@pytest.mark.parametrize(
    ("emitted", "probability", "reason", "token"),
    [
        (4, 1.0, STOP_REASON_CONTINUE, AUDIO_PATCH_TOKEN_ID),
        (5, 0.51, STOP_REASON_STOP_HEAD, AUDIO_END_TOKEN_ID),
        (800, 0.0, STOP_REASON_MAX_DECODE_STEPS, AUDIO_END_TOKEN_ID),
    ],
)
def test_stop_is_evaluated_after_emitting_latent(emitted, probability, reason, token):
    decision = resolve_stop_decision(
        emitted_steps=emitted,
        stop_probability=probability,
        stop_threshold=0.5,
        minimum_steps=5,
        maximum_steps=800,
        continue_token_id=AUDIO_PATCH_TOKEN_ID,
        stop_token_id=AUDIO_END_TOKEN_ID,
    )
    assert decision.reason == reason
    assert decision.next_token_id == token


def test_runtime_rejects_flow_vae_patch_confusion():
    config = _config()
    config.audio_config.patch_size = 1
    runtime = LingV3TTSRuntimeConfig.from_hf_config(config)
    with pytest.raises(ValueError, match="AudioVAE patch_size must be 4"):
        runtime.validate()


def test_runtime_rejects_incompatible_mla_geometry():
    config = _config()
    config.llm_config.kv_lora_rank = 256
    runtime = LingV3TTSRuntimeConfig.from_hf_config(config)
    with pytest.raises(ValueError, match="expected MLA head size 576"):
        runtime.validate()


def test_pipeline_contract():
    from vllm_omni.model_executor.models.lingv3_tts.pipeline import (
        LINGV3_TTS_PIPELINE,
    )

    stage0, stage1 = LINGV3_TTS_PIPELINE.stages
    assert stage0.hf_config_name == "llm_config"
    assert stage0.owns_tokenizer is True
    assert stage0.sampling_constraints["max_tokens"] == 801
    assert stage1.final_output is True
    assert stage1.final_output_type == "audio"
