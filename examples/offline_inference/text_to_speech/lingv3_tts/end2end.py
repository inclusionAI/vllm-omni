# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import json
import os
import re
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from vllm_omni import Omni
from vllm_omni.model_executor.models.lingv3_tts.constants import (
    AUDIO_END_TOKEN_ID,
    KEY_MAX_DECODE_STEPS,
    KEY_TEMPERATURE,
    MAX_DECODE_STEPS,
)
from vllm_omni.model_executor.models.lingv3_tts.prompt_assembly import build_lingv3_prompt
from vllm_omni.model_executor.models.ming_tts.prompt_assembly import create_instruction

DEPLOY_CONFIG = Path(__file__).resolve().parents[4] / "vllm_omni/deploy/lingv3_tts.yaml"
TASKS = (
    "seed",
    "multilingual",
    "podcast",
    "basic",
    "emotion",
    "emotion_zeroshot",
    "dialect",
    "ip",
    "voice_design",
    "freeform",
    "paralinguistic",
    "emergent",
)
REFERENCE_TEXT_TASKS = {"seed", "multilingual", "podcast", "paralinguistic"}
NO_REFERENCE_TASKS = {"emotion_zeroshot", "ip", "voice_design", "emergent"}
CONTROL_FIELDS = {
    "basic": {"speed": "语速", "pitch": "基频", "volume": "音量"},
    "emotion": {"emotion": "情感"},
    "emotion_zeroshot": {"emotion": "情感"},
    "dialect": {"dialect": "方言"},
    "ip": {"ip": "IP"},
    "voice_design": {"style": "风格"},
}
CONTROL_NAMES = {"speed", "pitch", "volume", "emotion", "dialect", "ip", "style", "instruction"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run LingV3-Tiny TTS offline with vLLM-Omni.")
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--model", required=True, help="Local LingV3-Tiny model directory")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", required=True, type=Path, help="Output WAV path")
    parser.add_argument(
        "--ref-audio",
        "--prompt-wav",
        dest="ref_audio",
        action="append",
        type=Path,
        help="Reference WAV; repeat for Podcast",
    )
    parser.add_argument(
        "--ref-text",
        "--prompt-text",
        dest="ref_text",
        help="Reference transcript; for Podcast, use ordered speaker_N: lines",
    )
    parser.add_argument("--language", "--target-language", dest="language")
    parser.add_argument("--speed", choices=("慢速", "中速", "快速"))
    parser.add_argument("--pitch", choices=("低", "中", "高"))
    parser.add_argument("--volume", choices=("低", "中", "高"))
    parser.add_argument("--emotion", choices=("中性", "悲伤", "愤怒", "高兴", "惊讶", "恐惧", "厌恶"))
    parser.add_argument("--dialect", choices=("川渝话", "广粤话"))
    parser.add_argument("--ip")
    parser.add_argument("--style")
    parser.add_argument("--instruction")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-decode-steps", type=int, help="Maximum audio decode steps")
    parser.add_argument("--deploy-config", type=Path, default=DEPLOY_CONFIG)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text must not be empty")
    if args.max_decode_steps is not None and not 5 <= args.max_decode_steps <= MAX_DECODE_STEPS:
        parser.error(f"--max-decode-steps must be between 5 and {MAX_DECODE_STEPS}")
    if args.task in NO_REFERENCE_TASKS:
        if args.ref_audio:
            parser.error(f"{args.task} does not accept reference audio")
    elif args.task == "podcast":
        if not args.ref_audio or len(args.ref_audio) < 2:
            parser.error("podcast requires at least two --ref-audio inputs")
    elif not args.ref_audio or len(args.ref_audio) != 1:
        parser.error(f"{args.task} requires exactly one --ref-audio")
    if args.task in REFERENCE_TEXT_TASKS:
        if not args.ref_text:
            parser.error(f"{args.task} requires --ref-text")
    elif args.ref_text is not None:
        parser.error(f"{args.task} does not accept --ref-text")
    languages = {
        "seed": {"Chinese", "English"},
        "multilingual": {"Japanese", "Korean", "German", "French", "Spanish"},
        "paralinguistic": {"Chinese", "English"},
    }
    allowed_languages = languages.get(args.task)
    if allowed_languages and args.language not in allowed_languages:
        parser.error(f"{args.task} requires --language in {', '.join(sorted(allowed_languages))}")
    if allowed_languages is None and args.language is not None:
        parser.error(f"{args.task} does not accept --language")
    expected_controls = set(CONTROL_FIELDS.get(args.task, {}))
    if args.task == "freeform":
        expected_controls = {"instruction"}
    supplied_controls = {name for name in CONTROL_NAMES if getattr(args, name) is not None}
    if supplied_controls != expected_controls:
        parser.error(
            f"{args.task} requires {', '.join(sorted(expected_controls)) or 'no control fields'}; "
            f"got {', '.join(sorted(supplied_controls)) or 'none'}"
        )
    if any(not getattr(args, name).strip() for name in supplied_controls):
        parser.error("Control text must not be empty")
    if args.task == "podcast":
        expected = [str(index) for index in range(1, len(args.ref_audio) + 1)]
        ref_speakers = re.findall(r"(?m)^\s*speaker_([1-9][0-9]*):\s*\S", args.ref_text)
        text_lines = [line for line in args.text.splitlines() if line.strip()]
        text_matches = [re.fullmatch(r"\s*speaker_([1-9][0-9]*):\s*\S.*", line) for line in text_lines]
        if (
            ref_speakers != expected
            or any(match is None for match in text_matches)
            or {match[1] for match in text_matches} != set(expected)
        ):
            parser.error("Use speaker_1 ... speaker_N in --text and ordered --ref-text")
    return args


def build_instruction(args):
    if args.task == "freeform":
        value = args.instruction
        if value.startswith("Speaking instruction:"):
            return value
        return f"Speaking instruction: Speak in this style: {value}"
    fields = CONTROL_FIELDS.get(args.task)
    if fields is None:
        return None
    return create_instruction({label: getattr(args, name) for name, label in fields.items()})


def load_references(paths):
    waveforms, sample_rates = [], []
    for path in paths or []:
        samples, rate = sf.read(path, dtype="float32")
        waveform = torch.as_tensor(samples, dtype=torch.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)
        waveform = waveform.reshape(1, -1)
        if waveform.numel() == 0:
            raise ValueError(f"Reference WAV is empty: {path}")
        waveforms.append(waveform)
        sample_rates.append(int(rate))
    return (waveforms or None), (sample_rates or None)


def main():
    args = parse_args()
    instruction = build_instruction(args)
    tts_config = json.loads((Path(args.model) / "config.json").read_text(encoding="utf-8")).get("tts_config") or {}
    max_steps = args.max_decode_steps
    if max_steps is None and args.task != "podcast":
        max_steps = int(tts_config.get("maximum_decode_steps", MAX_DECODE_STEPS))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False, fix_mistral_regex=True)
    waveforms, sample_rates = load_references(args.ref_audio)
    prompt = build_lingv3_prompt(
        tokenizer,
        target_text=args.text,
        reference_waveform=waveforms,
        reference_sample_rate=sample_rates,
        prompt_text=args.ref_text,
        target_language=args.language,
        instruction=instruction,
        max_decode_steps=max_steps,
        speaker_slot=args.task != "emergent",
    )
    prompt["additional_information"]["tts_local_seed"] = args.seed
    max_steps = prompt["additional_information"][KEY_MAX_DECODE_STEPS]
    sampling = [
        SamplingParams(temperature=0.0, max_tokens=max_steps + 1, stop_token_ids=[AUDIO_END_TOKEN_ID]),
        SamplingParams(temperature=0.0, max_tokens=1),
    ]
    prompt["additional_information"][KEY_TEMPERATURE] = sampling[0].temperature
    engine = Omni(model=args.model, deploy_config=str(args.deploy_config), trust_remote_code=False)
    try:
        outputs = engine.generate(prompts=[prompt], sampling_params_list=sampling, use_tqdm=False)
        audio_outputs = [output for output in outputs if output.final_output_type == "audio"]
        if len(audio_outputs) != 1:
            raise RuntimeError(f"Expected one audio output, got {len(audio_outputs)}")
        data = audio_outputs[0].multimodal_output or {}
        audio = data.get("audio")
        if audio is None:
            raise RuntimeError("Model returned no audio")
        if isinstance(audio, list):
            audio = torch.cat([torch.as_tensor(part).reshape(-1) for part in audio])
        audio = torch.as_tensor(audio, dtype=torch.float32).reshape(-1)
        if audio.numel() == 0 or not torch.isfinite(audio).all():
            raise RuntimeError("Model returned empty or non-finite audio")
        sample_rate = data.get("sr")
        if isinstance(sample_rate, list):
            sample_rate = sample_rate[-1]
        if sample_rate is None:
            raise RuntimeError("Model returned no sample rate")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output, audio.clamp(-1, 1).cpu().numpy(), int(sample_rate), subtype="PCM_16")
        print(f"Saved {args.output} ({audio.numel() / int(sample_rate):.2f}s, {int(sample_rate)} Hz)")
    finally:
        engine.close()


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
