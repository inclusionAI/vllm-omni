# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference and IP voice selection shared by completed and chunked speech."""

import json
import os
from pathlib import Path
from typing import Any

from .prompt_utils import create_instruction, infer_omni_language

_REFERENCE_DIR = Path(__file__).parent / "data"
_DEFAULT_IP = {"Chinese": "灵小柔", "English": "US_Stella"}


def resolve_omni_voice(text: str, controls: dict[str, Any]) -> dict[str, Any]:
    mode = controls.get("voice_mode") or "reference"
    if mode not in {"reference", "ip"}:
        raise ValueError("voice_mode must be 'reference' or 'ip'")
    if mode == "ip":
        name = controls.get("ip_name") or _DEFAULT_IP[infer_omni_language(text)]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("ip_name must be a nonempty string")
        # An IP supplies voice identity through the caption. References and
        # cached features from an earlier request must not affect that voice.
        return {
            "voice_mode": mode,
            "ip_name": name,
            "instruction": create_instruction({"IP": name}),
            "speaker_conditioning": "mute",
            "prompt_audio_mode": "none",
            "use_zero_spk_emb": False,
            **dict.fromkeys(
                (
                    "prompt_wav",
                    "reference_audio",
                    "prompt_text",
                    "spk_emb",
                    "prompt_wav_lat",
                    "prompt_wav_emb",
                    "voice_name",
                )
            ),
        }

    reference = {key: controls.get(key) for key in ("prompt_wav", "reference_audio", "prompt_text", "voice_name")}
    has_reference = (
        bool(reference["prompt_wav"])
        or bool(reference["reference_audio"])
        or bool(reference["voice_name"])
        or controls.get("spk_emb") is not None
    )
    if not has_reference:
        override = os.getenv("MING_TTS_REFERENCE_WAV")
        if override:
            reference.update(prompt_wav=override, prompt_text=os.getenv("MING_TTS_REFERENCE_TEXT") or None)
        else:
            manifest = json.loads((_REFERENCE_DIR / "default_reference.json").read_text())
            reference.update(prompt_wav=str(_REFERENCE_DIR / manifest["filename"]), prompt_text=manifest["text"])
    audio_mode = controls.get("prompt_audio_mode") or (
        "continuation" if reference["prompt_text"] or reference["voice_name"] else "speaker_only"
    )
    if audio_mode not in {"continuation", "speaker_only"}:
        raise ValueError("reference voice_mode requires continuation or speaker_only prompt_audio_mode")
    return {
        **reference,
        "voice_mode": mode,
        "ip_name": None,
        "instruction": None,
        "speaker_conditioning": "prompt",
        "prompt_audio_mode": audio_mode,
        "use_zero_spk_emb": False,
    }
