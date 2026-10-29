# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import base64
from pathlib import Path

import httpx


def encode_reference(path):
    return "data:audio/wav;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def main():
    parser = argparse.ArgumentParser(description="Generate LingV3 speech through /v1/audio/speech.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default="lingv3-tts")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-audio", type=Path, action="append", help="Reference WAV; repeat for multiple speakers.")
    parser.add_argument("--ref-text")
    parser.add_argument("--instructions", help="Structured JSON or free-form style instruction.")
    parser.add_argument("--language")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.ref_text and not args.ref_audio:
        parser.error("--ref-text requires --ref-audio")
    payload = dict(model=args.model, input=args.text, seed=args.seed, response_format="wav")
    for key in ("ref_text", "instructions", "language"):
        value = getattr(args, key)
        if value is not None:
            payload[key] = value
    try:
        refs = [encode_reference(path) for path in args.ref_audio or []]
        if refs:
            payload["ref_audio"] = refs[0] if len(refs) == 1 else refs
        with httpx.Client(timeout=args.timeout, headers={"Authorization": f"Bearer {args.api_key}"}) as client:
            response = client.post(args.base_url.rstrip("/") + "/v1/audio/speech", json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        if not response.content:
            raise ValueError("Empty audio response")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(response.content)
        print(f"Saved {args.output}")
    except (OSError, httpx.HTTPError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
