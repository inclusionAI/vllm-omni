# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import unittest
from types import SimpleNamespace as S

import torch

from vllm_omni.model_executor.logits_processor import MoveDirectionality, NoRepeatNGramLogitsProcessor, banned_tokens


class NoRepeatTests(unittest.TestCase):
    def test_prompt_history_and_whitelist(self):
        self.assertEqual(banned_tokens([1, 2, 3, 9, 1, 2], 3, 8192), {3})
        self.assertEqual(banned_tokens([1, 2, 3, 9, 1, 2], 3, 8192, [3]), set())
        self.assertEqual(banned_tokens([1, 2, 3, 9, 1, 2], 3, 2), set())
        self.assertEqual(banned_tokens([1, 2], 0, 8192), set())

    def test_live_output_and_batch_moves(self):
        p = NoRepeatNGramLogitsProcessor(None, None, False)
        output = []
        params = S(extra_args={"no_repeat_ngram_size": 3, "no_repeat_ngram_window": 8192})
        p.update_state(
            S(removed=[], added=[(0, params, [1, 2, 3, 8], output), (1, S(extra_args={}), [], [])], moved=[])
        )
        output.extend([1, 2])
        x = torch.zeros(2, 10)
        p.apply(x)
        self.assertTrue(torch.isneginf(x[0, 3]))
        self.assertTrue(torch.isfinite(x[1]).all())
        p.update_state(S(removed=[], added=[], moved=[(0, 1, MoveDirectionality.SWAP)]))
        x = torch.zeros(2, 10)
        p.apply(x)
        self.assertTrue(torch.isneginf(x[1, 3]))
        self.assertTrue(torch.isfinite(x[0]).all())
        p.update_state(S(removed=[0], added=[], moved=[(1, 0, MoveDirectionality.UNIDIRECTIONAL)]))
        self.assertIs(p.requests[0][1], output)
        p.update_state(S(removed=[0], added=[], moved=[]))
        self.assertEqual(p.requests, {})

    def test_ten_token_repetition(self):
        seq = [7] * 10
        self.assertEqual(banned_tokens(seq, 10, 8192), {7})
        self.assertEqual(banned_tokens(seq[:9], 10, 8192), set())


if __name__ == "__main__":
    unittest.main()
