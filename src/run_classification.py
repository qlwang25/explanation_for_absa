# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
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
import os
import sys
sys.path.append('../')
sys.path.append("../../")
import logging
import argparse
import random
import json
import math
from tqdm import tqdm
from collections import namedtuple
import pickle

import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from ema_pytorch import EMA

from transformer_utils.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformer_utils.models.t5.modeling_t5 import T5PreTrainedModel, T5ForConditionalGeneration


from optimization import AdamW, WarmupLinearSchedule, Warmup
from utils import ABSCTokenizer, ABSCProcessor
from colorlogging import getLogger
import modelconfig
# from AEN import AENForClassification, convert_examples_to_features_cls
# from BERT import BERTForClassification, convert_examples_to_features_cls
# from Scon import SconForClassification, convert_examples_to_features_cls


logger = getLogger(__name__)
label2id = {'negative':0, 'neutral':1, 'positive':2}

class BertForClassification(BertPreTrainedModel):
    def __init__(self, config, clue_num=0, num_labels=3):
        super(BertForClassification, self).__init__(config)
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


class T5ForClassification(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super(T5ForClassification, self).__init__()
        self.t5 = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        if labels is not None:
            labels[labels == 0] = -100
            outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
            loss = outputs.loss
            pooled_output = outputs.decoder_hidden_states[-1][:, 1, :]
            logits = outputs.logits[:, 1, :]
            return loss, pooled_output, logits
        else:
            outputs = self.t5.generate(input_ids=input_ids, attention_mask=attention_mask)
            predicts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True) # ['negative', 'positive', 'negative', 'positive', 'neutral']
            predicts = torch.tensor([label2id[p] for p in predicts]).cuda()
            return predicts


def train(args, trainset, testset, model):
    train_dataloader = DataLoader(trainset, sampler=RandomSampler(trainset), batch_size=args.train_batch_size)
    num_train_steps = int(len(train_dataloader)) * args.num_train_epochs

    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.warmup_proportion)
    scheduler = Warmup[args.schedule](optimizer, warmup_steps=args.warmup_steps, t_total=num_train_steps)

    logger.info("Total optimization steps = %d", num_train_steps)
    model.zero_grad()
    model.train()
    global_step = 0
    # update_after_step = 150
    # ema = EMA(model, beta = 0.95, update_after_step = update_after_step, update_every = 5,)
    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_masks, input_seg_ids, output_ids, output_masks, output_seg_ids, label_ids = batch 

            loss1, pooled_output1, logit1 = model(input_ids, token_type_ids=input_seg_ids, attention_mask=input_masks, labels=label_ids,)

            loss2, loss3 = torch.tensor([0]).cuda(), torch.tensor([0]).cuda()
            # if global_step > update_after_step:
            #     with torch.no_grad():
            #         _, pooled_output2, logit2 = ema(output_ids, token_type_ids=output_seg_ids, attention_mask=output_masks, labels=label_ids,)
            #     loss2 = torch.nn.KLDivLoss(reduction="batchmean")(input=F.log_softmax(logit1, dim=1), target=F.softmax(logit2, dim=1))
            #     loss3 = torch.nn.MSELoss()(input=pooled_output1, target=pooled_output2)

            # (loss1 + 0.1*loss2 + 0.1*loss3).backward()    
            (loss1).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            # ema.update()

            loss2, _, _ = model(output_ids, token_type_ids=output_seg_ids, attention_mask=output_masks, labels=label_ids,)
            (loss2).backward() #       
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1
            if global_step % args.logging_global_step == 0:
                logger.info("Epoch:{}, Global Step:{}/{}, Loss1:{:.5f}, Loss2:{:.5f}, Loss3:{:.5f}".format(epoch, global_step, num_train_steps, loss1.item(), loss2.item(), loss3.item()),)
                if args.evaluate_during_training and global_step % args.eval_logging_steps == 0: 
                    model.eval()
                    evaluate(args, testset=testset, model=model)
                    model.train()
            torch.cuda.empty_cache()


def evaluate(args, testset, model):
    eval_dataloader = DataLoader(testset, sampler=SequentialSampler(testset), batch_size=args.eval_batch_size)
    model.eval()
    out_preds, out_labes = [], []
    for batch in eval_dataloader:
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_masks, input_seg_ids, _, _, _, label_ids = batch 

            if args.plm_name == 't5':
                labels = args.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
                label_ids = torch.tensor([label2id[l] for l in labels]).cuda()

            predicts = model(input_ids, token_type_ids=input_seg_ids, attention_mask=input_masks,)
            out_preds.append(predicts.detach().cpu().numpy())
            out_labes.append(label_ids.detach().cpu().numpy())
        torch.cuda.empty_cache()
    
    y_true = np.concatenate(tuple(out_labes), axis=0)
    y_pred = np.concatenate(tuple(out_preds), axis=0)
    logger.info("accuracy: {:.4}; precision:{:.4}; recall:{:.4}; f1:{:.4}".format(accuracy_score(y_true, y_pred), 
        precision_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average='macro'),
        f1_score(y_true, y_pred, average='macro')),
    )
    np.save(os.path.join(args.data_dir, 'true.pt'), y_true)
    np.save(os.path.join(args.data_dir, 'pred.pt'), y_pred)


def load_and_cache_examples(args, dataname="train", key=None):
    processor = ABSCProcessor()
    logger.info("Creating features from dataset file at %s", args.data_dir)
    if dataname == "train":
        examples = processor.get_train_examples(args.data_dir)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", args.train_batch_size)
    elif dataname == "test":
        examples = processor.get_test_examples(args.data_dir)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
    else:
        raise ValueError("(evaluate and dataname) parameters error !")

    label_map = processor.get_labels()
    features = convert_examples_to_features_cls(examples, 
        label2id=label_map, 
        max_seq_length=args.max_seq_length,
        tokenizer=args.tokenizer, 
        plm_name=args.plm_name,
        printN=1,
        )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_masks for f in features], dtype=torch.long)
    all_input_seg_ids = torch.tensor([f.input_seg_ids for f in features], dtype=torch.long)

    all_output_ids = torch.tensor([f.output_ids for f in features], dtype=torch.long)
    all_output_masks = torch.tensor([f.output_masks for f in features], dtype=torch.long)
    all_output_seg_ids = torch.tensor([f.output_seg_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_masks, all_input_seg_ids, all_output_ids, all_output_masks, all_output_seg_ids, all_label_ids,)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--plm_model", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--schedule", default="WarmupLinearSchedule", type=str,
                        help="Can be `'WarmupLinearSchedule'`, `'warmup_constant'`, `'warmup_cosine'` , `'none'`,"
                             " `None`, 'warmup_cosine_warmRestart' or a `warmup_cosine_hardRestart`")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--logging_global_step', type=int, default=100, help="Log every X updates steps.")
    parser.add_argument('--eval_logging_steps', type=int, default=300, help="Log every X evalution steps.")
    parser.add_argument("--evaluate_during_training", action='store_true')
    parser.add_argument("--clue_num", default=0, type=int)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Instance = ABSCTokenizer(plm_name=args.plm_model, archive=modelconfig.MODEL_ARCHIVE_MAP[args.plm_model])
    args.tokenizer = Instance.tokenizer
    args.plm_name = Instance.name
    
    dataset = load_and_cache_examples(args, dataname="train")


    model = BERTForClassification.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.plm_model])

    # t5 = T5ForConditionalGeneration.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.plm_model])
    # t5.cuda()
    # model = T5ForClassification(model=t5, tokenizer=args.tokenizer)

    # model = AENForClassification.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.plm_model])
    # model = SSEGCNBERTForClassification.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.plm_model])
    model.cuda()

    if args.do_train:
        train(args, trainset=dataset, testset=None, model=model)

    for key in ["implicit", "explicit"]:
        logger.info("**********{}*************".format(key))
        eval_data = load_and_cache_examples(args, dataname="test", key=key)
        if args.do_eval:
            evaluate(args, testset=eval_data, model=model)

if __name__ == "__main__":
    main()
