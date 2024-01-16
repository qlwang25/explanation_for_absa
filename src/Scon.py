# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('../')
sys.path.append("../../")

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_utils.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from utils import InputFeatures

import re
from colorlogging import getLogger
from collections import defaultdict
from copy import deepcopy

logger = getLogger(__name__)


def convert_examples_to_features_cls(examples, 
    label2id, 
    max_seq_length, 
    tokenizer,
    mask_padding_with_zero=True,
    plm_name='bert',
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
        input_ids, input_masks, input_seg_ids = inputIdMaskSegment(tmp_tokens=example.text.lower(), tmp_aspect=example.aspect.lower(), segment=0)
        output_ids, output_masks, output_seg_ids = inputIdMaskSegment(tmp_tokens=example.explain.lower(), tmp_aspect=example.aspect.lower(), segment=1)
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


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)     # 16*1
            mask = torch.eq(labels, labels.T).float()      # 16*16

        features = features.unsqueeze(dim=1)
        features = F.normalize(features, dim=2)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = features[:, 0]

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_min, _ = torch.min(logits, dim=1, keepdim=True)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = torch.div(logits-logits_min, logits_max - logits_min)

        logits_mask = 1 - torch.eye(batch_size).to(labels.device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        return loss


class SconForClassification(BertPreTrainedModel):
    def __init__(self, config, clue_num=0, num_labels=3):
        super(SconForClassification, self).__init__(config)
        config.clue_num = clue_num
        hidden_size = config.hidden_size

        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.info = SupConLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        _, pool_out = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=False,)
        pool_out = self.dropout(pool_out)
        logits = self.classifier(pool_out)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            conloss = self.info(pool_out, labels)
            return loss + conloss, pool_out, logits
        else:
            probability = torch.nn.Softmax(dim=1)(logits)
            predicts = torch.max(probability, dim=1)[1]
            return predicts
