# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
sys.path.append("../../")

import re
import math
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
    printN=1,):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    mask_token = tokenizer.mask_token
    cls_token_id = tokenizer.convert_tokens_to_ids([cls_token])[0]
    sep_token_id = tokenizer.convert_tokens_to_ids([sep_token])[0]
    pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]

    def inputIdMaskSegment(tmp_tokens=None, segment=None, tmp_seq_length=max_seq_length):
        text = re.sub('|'.join(r'\b' + kw + r'\b' for kw in ['negative', 'neutral', 'positive']), '[MASK]', tmp_tokens, flags=re.IGNORECASE)

        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [cls_token_id] + input_ids[:tmp_seq_length-2] + [sep_token_id]
        
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = tmp_seq_length - len(input_ids)
        input_ids += ([pad_token_id] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = [segment] * tmp_seq_length
        assert len(input_ids) == len(input_mask) == len(segment_ids) == tmp_seq_length
        return input_ids, input_mask, segment_ids

    features = []
    Segment = {'bert': inputIdMaskSegment}
    for (ex_index, example) in enumerate(examples):
        if ex_index % 2000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        label_id = label2id[example.label]
        text = example.text.lower()
        aspect = example.aspect.lower()
        explain = example.explain.lower()
        input_ids, input_masks, input_seg_ids = Segment[plm_name](tmp_tokens=text, segment=0)
        output_ids, output_masks, output_seg_ids = Segment[plm_name](tmp_tokens=explain, segment=0)
        aspect_ids, aspect_masks, aspect_seg_ids = Segment[plm_name](tmp_tokens=aspect, segment=0, tmp_seq_length=15)
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


class Attention(torch.nn.Module):
    def __init__(self, input_dim=None, out_dim=None, n_head=1, dropout=0.1):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim // n_head
        self.n_head = n_head
        self.w_k = nn.Linear(input_dim, n_head * self.hidden_dim)
        self.w_q = nn.Linear(input_dim, n_head * self.hidden_dim)
        self.proj = nn.Linear(n_head * self.hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(self.hidden_dim*2))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        mb_size, k_len, _ = k.size()
        q_len = q.shape[1]

        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)

        kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
        qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
        kq = torch.cat((kxx, qxx), dim=-1)
        score = torch.tanh(torch.matmul(kq, self.weight))
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_hid, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output


class AENForClassification(BertPreTrainedModel):
    def __init__(self, config, clue_num=0, num_labels=3):
        super(AENForClassification, self).__init__(config)
        config.clue_num = clue_num
        self.config = config
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.attn_k = Attention(input_dim=config.hidden_size, out_dim=config.hidden_size // 3, n_head=8, dropout=config.hidden_dropout_prob)
        self.attn_q = Attention(input_dim=config.hidden_size, out_dim=config.hidden_size // 3, n_head=8, dropout=config.hidden_dropout_prob)
        self.ffn_c = PositionwiseFeedForward(d_hid=config.hidden_size // 3, dropout=config.hidden_dropout_prob)
        self.ffn_t = PositionwiseFeedForward(d_hid=config.hidden_size // 3, dropout=config.hidden_dropout_prob)
        self.attn_s1 = Attention(input_dim=config.hidden_size // 3, out_dim=config.hidden_size // 3, n_head=8, dropout=config.hidden_dropout_prob)

        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, token_type_ids, attention_mask, aspect_ids, aspect_seg_ids, aspect_masks, labels=None):
        context_len = torch.sum(attention_mask != 0, dim=-1)
        target_len = torch.sum(aspect_masks != 0, dim=-1)

        context, _ = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=False,)
        context = self.dropout(context)
        target, _ = self.bert(aspect_ids, token_type_ids=aspect_seg_ids, attention_mask=aspect_masks, output_all_encoded_layers=False,)
        target = self.dropout(target)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.unsqueeze(1).float())
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.unsqueeze(1).float())
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.unsqueeze(1).float())

        pooled_output = torch.cat([hc_mean, s1_mean, ht_mean], dim=-1)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return loss, pooled_output, logits
        else:
            probability = torch.nn.Softmax(dim=1)(logits)
            predicts = torch.max(probability, dim=1)[1]
            return predicts