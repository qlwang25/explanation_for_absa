# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import csv
import os
import sys
sys.path.append("../")
sys.path.append("../../")
import json
import re
import random
import pickle
from collections import defaultdict
import numpy as np
from random import shuffle

import torch
from torch.utils.data import Dataset
from transformer_utils.models.bert.tokenization_bert import BertTokenizer
from transformer_utils.models.t5.tokenization_t5 import T5Tokenizer


from colorlogging import getLogger
logger = getLogger(__name__)


class ABSCTokenizer(object):
    def __init__(self, plm_name, archive=None):
        if 'bert_' in plm_name:
            self.tokenizer = BertTokenizer.from_pretrained(archive)
            self.name = 'bert'
        elif 't5_' in plm_name:
            self.tokenizer = T5Tokenizer.from_pretrained(archive)
            self.name = 't5'
        elif 'gpt' in plm_name:
            self.tokenizer = GPT2Tokenizer.from_pretrained(archive)
            self.name = 'gpt'
        else:
            raise ValueError


class InputExample(object):
    def __init__(self, guid, text, aspect=None, label=None, explain=None):
        self.guid = guid
        self.text = text
        self.aspect = aspect
        self.label = label
        self.explain = explain


class InputFeatures(object):
    def __init__(self, 
        input_ids=None, input_masks=None, input_seg_ids=None, 
        output_ids=None, output_masks=None, output_seg_ids=None, 
        aspect_ids=None, aspect_masks=None, aspect_seg_ids=None,
        label_id=None,
        ):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.input_seg_ids = input_seg_ids
        self.output_ids = output_ids
        self.output_masks = output_masks
        self.output_seg_ids = output_seg_ids
        self.aspect_ids = aspect_ids
        self.aspect_masks = aspect_masks
        self.aspect_seg_ids = aspect_seg_ids
        self.label_id = label_id


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    def clean_text(text):
        text = re.sub(r'\"{2,}', '"', text)
        text = re.sub(r'\`{2,}', "", text)
        text = re.sub(r'\'{2,}', "'", text)
        return text
    @classmethod
    def _read_txt(cls, input_path, quotechar=None):
        lines = []
        with open(input_path, 'r') as fread:
            logger.info("load file {}".format(input_path))
            if 'train' in input_path:
                for line in fread.readlines():
                    sent, aspect, label, explain = line.strip().split('|||')
                    sent = re.sub(r"(?<=\w)([\.\?\!\,\:\;\-\)])", r" \1", sent)
                    explain = re.sub(r"(?<=\w)([\.\?\!\,\:\;\-\)])", r" \1", explain)
                    explain = explain.lstrip()
                    lines.append((cls.clean_text(sent), cls.clean_text(aspect), label, cls.clean_text(explain)))
            else:
                while True:
                    sent = fread.readline().strip()
                    if not sent:
                        break
                    aspect = fread.readline().strip()
                    label = fread.readline().strip()
                    sent = sent.replace('$T$', aspect)

                    sent = cls.clean_text(sent)
                    aspect = cls.clean_text(aspect)
                    for k, v in {" 's":"'s", " n't":"n't", " 've":"'ve", " %":"%", " '":"'", " 're":"'re", " 'm":"'m", "''":"'"}.items():
                        sent = sent.replace(k, v)
                        aspect = aspect.replace(k, v)
                    lines.append((sent, aspect, label, "NONE"))
        return lines

class ABSCProcessor_cls(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_txt(os.path.join(data_dir, "train.pred")), set_type="train")
    def get_test_examples(self, data_dir):
        lines = self._read_txt(os.path.join(data_dir, "test.raw"))
        return self._create_examples(lines, set_type="test")
    def get_labels(self):
        return {'-1':0, '0':1, '1':2}
    def _create_examples(self, lines, set_type):
        examples = []
        for i, (text, aspect, label, explain) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text=text, aspect=aspect, label=label, explain=explain))
        return examples


def convert_examples_to_features_cls(examples, 
    label2id, 
    max_seq_length, 
    tokenizer,
    mask_padding_with_zero=True,
    plm_name='bert',
    printN=1,):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    mask_token = tokenizer.mask_token
    # eos_token = tokenizer.eos_token
    cls_token_id = tokenizer.convert_tokens_to_ids([cls_token])[0]
    sep_token_id = tokenizer.convert_tokens_to_ids([sep_token])[0]
    pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]
    # eos_token_id = tokenizer.convert_tokens_to_ids([eos_token])[0]

    def inputIdMaskSegment(tmp_tokens=None, tmp_aspect=None, segment=None, tmp_seq_length=max_seq_length):
        tokens = tokenizer.tokenize(tmp_tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if tmp_aspect is not None:
            apt_tokens = tokenizer.tokenize(tmp_aspect.lower())
            apt_input_ids = tokenizer.convert_tokens_to_ids(apt_tokens)
            input_ids = [cls_token_id] + input_ids[:tmp_seq_length-3-len(apt_input_ids)] + [sep_token_id] + apt_input_ids + [sep_token_id]
        else:
            input_ids = [cls_token_id] + input_ids[:tmp_seq_length-2] + [sep_token_id]
        
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = tmp_seq_length - len(input_ids)
        input_ids += ([pad_token_id] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = [segment] * tmp_seq_length
        assert len(input_ids) == len(input_mask) == len(segment_ids) == tmp_seq_length
        return input_ids, input_mask, segment_ids

    def inputIdMaskSegmentT5(tmp_tokens=None, tmp_aspect=None, segment=None, tmp_seq_length=max_seq_length):
        input_ids = tokenizer([tmp_tokens], padding="longest", truncation=True)['input_ids'][0]
        if tmp_aspect is not None:
            apt_input_ids = tokenizer(["The sentiment of aspect {} is <extra_id_0> .".format(tmp_aspect.lower())], padding="longest", truncation=True)['input_ids'][0]
            input_ids = input_ids[:tmp_seq_length-1-len(apt_input_ids)] + [eos_token_id] + apt_input_ids
        else:
            # prompt_input_ids = tokenizer(["The sentiment is <extra_id_0> ."], padding="longest", truncation=True)['input_ids'][0]
            # input_ids = input_ids[:tmp_seq_length - 1 - len(prompt_input_ids)] + [eos_token_id] + prompt_input_ids
            input_ids = [eos_token_id] + input_ids[:tmp_seq_length - 1]
        
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = tmp_seq_length - len(input_ids)
        input_ids += ([pad_token_id] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = [segment] * tmp_seq_length
        assert len(input_ids) == len(input_mask) == len(segment_ids) == tmp_seq_length
        return input_ids, input_mask, segment_ids

    def reaplaceText(text, aspect, prob):
        text = text.replace(aspect, "$T$")
        words = text.lower().split()
        for i in range(len(words)):
            if words[i] in ['positive', 'negative', 'neutral']:
                words[i] = mask_token
                pass
            elif random.random() < prob and words[i] != '$t$':
                words[i] = mask_token
        new_text = ' '.join(words)
        return new_text

    features = []
    Segment = {'bert': inputIdMaskSegment, 't5': inputIdMaskSegmentT5, 'gpt': inputIdMaskSegmentGPT}
    for (ex_index, example) in enumerate(examples):
        if ex_index % 2000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        if plm_name == 'bert': 
            label_id = label2id[example.label]
        if plm_name == 't5':
            label2tag = {'-1':'negative', '0':'neutral', '1':'positive'}
            label_id = tokenizer(["<extra_id_0> {} <extra_id_1>".format(label2tag[example.label])], padding="longest", truncation=True)['input_ids'][0]
        if plm_name == 'gpt':
            label_id = label2id[example.label]

        text = reaplaceText(text=example.text, aspect=example.aspect, prob=0.0)
        input_ids, input_masks, input_seg_ids = Segment[plm_name](tmp_tokens=text, tmp_aspect=None, segment=0)
        explain = reaplaceText(text=example.explain, aspect=example.aspect, prob=0.0)
        output_ids, output_masks, output_seg_ids = Segment[plm_name](tmp_tokens=explain, tmp_aspect=None, segment=1)
        aspect_ids, aspect_masks, aspect_seg_ids = Segment[plm_name](tmp_tokens=example.aspect, tmp_aspect=None, segment=0, tmp_seq_length=15)
        features.append(InputFeatures(input_ids=input_ids, 
            input_masks=input_masks, 
            input_seg_ids=input_seg_ids,
            output_ids=output_ids,
            output_masks=output_masks,
            output_seg_ids=output_seg_ids,
            aspect_ids=aspect_ids,
            aspect_masks=aspect_masks,
            aspect_seg_ids=aspect_seg_ids,
            label_id=label_id,)
        )

        if ex_index < printN:
            logger.info("*** Example ***")
            logger.info("REVIEW: %s, And EXPLAIN: %s " % (text, explain))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_masks: %s" % " ".join([str(x) for x in input_masks]))
            logger.info("output_ids: %s" % " ".join([str(x) for x in output_ids]))
            logger.info("output_masks: %s" % " ".join([str(x) for x in output_masks]))
            logger.info("aspect_ids: %s" % " ".join([str(x) for x in aspect_ids]))
            logger.info("aspect_masks: %s" % " ".join([str(x) for x in aspect_masks]))
            logger.info("label_ids: {}".format(label_id))
    return features
