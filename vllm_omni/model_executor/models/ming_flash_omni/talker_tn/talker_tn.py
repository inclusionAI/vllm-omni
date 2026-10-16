# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Email: wanren.pj@antgroup.com
# Date: 2025/10/16
import re
import os
import logging
from .talker_pre_processor import TalkerPreProcessor
from .normalizer import Normalizer
from .talker_re import TalkerRE


class TalkerTN:
    def __init__(self):
        self.talker_pre_processor = TalkerPreProcessor()
        self.talker_re = TalkerRE()
        file_dir = os.path.dirname(os.path.realpath(__file__))
        default_re_cfg = f'{file_dir}/talker_re.json'
        if os.path.exists(default_re_cfg):
            self.talker_re.update(default_re_cfg)
        self.re_contains_chinese = re.compile(r'[一-鿿]')
        self.re_digit_hyphen = re.compile(r'(?<=\d)-(?=\d)')
        self.tn_zh = Normalizer(f'{file_dir}/zh_tn', ordertype='tn')
        self.tn_en = Normalizer(f'{file_dir}/en_tn', ordertype='en_tn')

    def __call__(self, text):
        logging.debug(f"[TalkerTN] input: {text!r}")
        text = self.talker_pre_processor(text)
        logging.debug(f"[TalkerTN] after pre_processor: {text!r}")
        if len(text) == 0:
            return text

        is_chinese = bool(self.re_contains_chinese.search(text))
        logging.debug(f"[TalkerTN] is_chinese={is_chinese}")
        if is_chinese:
            bak_text = text
            try:
                text = self.talker_re(text)
                logging.debug(f"[TalkerTN] after talker_re: {text!r}")
            except Exception as e:
                logging.debug(f"[TalkerTN] talker_re raised {e!r}, fallback to bak_text")
                text = bak_text
            text = self.re_digit_hyphen.sub('，', text)
            out = self.tn_zh.normalize(text)
            logging.debug(f"[TalkerTN] after tn_zh.normalize: {out!r}")
            return out
        out = self.tn_en.normalize(text)
        logging.debug(f"[TalkerTN] after tn_en.normalize: {out!r}")
        return out

    def normalize(self, text):
        return self(text)
