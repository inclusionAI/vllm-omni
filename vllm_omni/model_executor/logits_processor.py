# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request-controlled logits processors."""

from vllm.v1.sample.logits_processor.interface import (
    LogitsProcessor,
    MoveDirectionality,
)


def banned_tokens(sequence, n, window, whitelist=()):
    if n <= 0 or window <= 0 or len(sequence) < n:
        return set()
    prefix = sequence[-(n - 1) :] if n > 1 else []
    banned = {
        sequence[i + n - 1]
        for i in range(max(0, len(sequence) - window), len(sequence) - n + 1)
        if n == 1 or sequence[i : i + n - 1] == prefix
    }
    return banned.difference(whitelist)


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Prevent repeated n-grams across prompt and generated tokens.

    Configure via SamplingParams.extra_args: no_repeat_ngram_size (default 0,
    disabled), no_repeat_ngram_window (default 8192), and
    no_repeat_ngram_whitelist (token IDs exempt from the restriction).
    """

    def __init__(self, vllm_config, device, is_pin_memory):
        self.requests = {}

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.requests.pop(index, None)
        for index, params, prompt_ids, output_ids in batch_update.added:
            extras = params.extra_args or {}
            self.requests[index] = (
                prompt_ids or [],
                output_ids,
                int(extras.get("no_repeat_ngram_size", 0)),
                int(extras.get("no_repeat_ngram_window", 8192)),
                extras.get("no_repeat_ngram_whitelist", []),
            )
        for source, target, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                self.requests[source], self.requests[target] = (self.requests[target], self.requests[source])
            else:
                self.requests[target] = self.requests.pop(source)

    def apply(self, logits):
        for index, (prompt, output, n, window, whitelist) in self.requests.items():
            banned = banned_tokens(list(prompt) + list(output), n, window, whitelist)
            if banned:
                logits[index, list(banned)] = -float("inf")
        return logits
