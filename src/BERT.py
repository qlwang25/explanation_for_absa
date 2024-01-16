# -*- coding: utf-8 -*-

import os
import re
import sys
sys.path.append('../')
sys.path.append("../../")

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_utils.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from utils import InputFeatures

from colorlogging import getLogger

logger = getLogger(__name__)

def convert_examples_to_features_cls(examples, 
    label2id, 
    max_seq_length, 
    tokenizer,
    mask_padding_with_zero=True,
    plm_name='bert',
    data_dir=None,
    printN=1,
    ):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    cls_token_id = tokenizer.convert_tokens_to_ids([cls_token])[0]
    sep_token_id = tokenizer.convert_tokens_to_ids([sep_token])[0]
    pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]

    def inputIdMaskSegment(tmp_tokens=None, tmp_aspect=None, segment=None, tmp_seq_length=max_seq_length):
        text = re.sub('|'.join(r'\b' + kw + r'\b' for kw in ['negative', 'neutral', 'positive']), '[MASK]', tmp_tokens, flags=re.IGNORECASE)
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        apt_tokens = tokenizer.tokenize(tmp_aspect)
        apt_input_ids = tokenizer.convert_tokens_to_ids(apt_tokens)
        input_ids = [cls_token_id] + input_ids[:tmp_seq_length-3-len(apt_input_ids)] + [sep_token_id] + apt_input_ids + [sep_token_id]
        segment_ids = [segment] * len(input_ids)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = tmp_seq_length - len(input_ids)
        input_ids += ([pad_token_id] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += [0] * padding_length
        assert len(input_ids) == len(input_mask) == len(segment_ids) == tmp_seq_length
        return input_ids, input_mask, segment_ids

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 2000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        label_id = label2id[example.label]
        text = example.text.lower()
        aspect = example.aspect.lower()
        input_ids, input_masks, input_seg_ids = inputIdMaskSegment(tmp_tokens=text, tmp_aspect=aspect, segment=0)
        explain = example.explain.lower()
        output_ids, output_masks, output_seg_ids = inputIdMaskSegment(tmp_tokens=explain, tmp_aspect=aspect, segment=1)
        features.append(InputFeatures(input_ids=input_ids, 
            input_masks=input_masks, 
            input_seg_ids=input_seg_ids,
            output_ids=output_ids,
            output_masks=output_masks,
            output_seg_ids=output_seg_ids,
            label_id=label_id,)
        )

        if ex_index < printN:
            logger.info("*** Example ***")
            logger.info("REVIEW: %s, And EXPLAIN: %s " % (example.text, example.explain))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_masks: %s" % " ".join([str(x) for x in input_masks]))
            logger.info("output_ids: %s" % " ".join([str(x) for x in output_ids]))
            logger.info("output_masks: %s" % " ".join([str(x) for x in output_masks]))
            logger.info("label_ids: {}".format(label_id))
    return features


class BERTForClassification(BertPreTrainedModel):
    def __init__(self, config, clue_num=0, num_labels=3):
        super(BERTForClassification, self).__init__(config)
        config.clue_num = clue_num
        self.config = config
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=False,)
        _, pooled_output = outputs 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return loss, pooled_output, logits
        else:
            probability = torch.nn.Softmax(dim=1)(logits)
            predicts = torch.max(probability, dim=1)[1]
            return predicts
