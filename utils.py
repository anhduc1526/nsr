import os
import json
import copy
import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics.v1 import _prf_divide
from abc import ABC, abstractmethod

def load_dataset(args, tokenizer, load_type):
    with open('./Datasets/' + args.dataset + '/' + args.num_unseen_labels + '/seed_' + str(args.seed) + '/' + load_type + '.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('./Datasets/' + args.dataset + '/' + args.dataset + '_prompts.json', 'r', encoding='utf-8') as f:
        descriptions = json.load(f)

    if args.dataset == 'wikizsl':
        dataset = WikiZSLDataset(args, data)
    elif args.dataset == 'fewrel':
        dataset = FewRelDataset(args, data)

    if load_type == 'train':
        train_collator = TrainCollator(args, tokenizer, descriptions)
        data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=train_collator, num_workers=args.num_workers)
    else:
        with open('./Datasets/' + args.dataset + '/' + args.num_unseen_labels + '/seed_' + str(args.seed) + '/' + load_type + '_labels.json', 'r', encoding='utf-8') as f:
            relations = json.load(f)

        eval_collator = EvalCollator(args, tokenizer, descriptions, relations)
        data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=True, collate_fn=eval_collator, num_workers=args.num_workers)

    return data_loader

def compute_macro_PRF(
        predicted_idx: np.ndarray, gold_idx: np.ndarray, i=-1, empty_label=None
):
    # https://github.com/dinobby/ZS-BERT/blob/master/model/evaluation.py
    """
    This evaluation function follows work from Sorokin and Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf)
    code borrowed from the following link:
    https://github.com/UKPLab/emnlp2017-relation-extraction/blob/master/relation_extraction/evaluation/metrics.py
    """
    if i == -1:
        i = len(predicted_idx)

    complete_rel_set = set(gold_idx) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        r_indices = predicted_idx[:i] == r
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
    f1 = 0.0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec + avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1

def add_entity_markers(sentence, head, tail):
    marked_sentence = sentence[:]

    h_start, h_end = head
    t_start, t_end = tail

    markers = [
        (h_start, '[HEAD]'),
        (h_end, '[/HEAD]'),
        (t_start, '[TAIL]'),
        (t_end, '[/TAIL]')
    ]

    markers.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for index, marker in markers:
        marked_sentence.insert(index, marker)

    return marked_sentence

class BaseDataset(Dataset):
    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_i = self.data[index]
        return {
            'sentence': add_entity_markers(data_i['sentence'], data_i['head'], data_i['tail']),
            'relation': data_i['relation'],
        }

class WikiZSLDataset(BaseDataset):
    def __init__(self, args, data):
        super().__init__(args, data)

class FewRelDataset(BaseDataset):
    def __init__(self, args, data):
        super().__init__(args, data)

class BaseCollator(ABC):
    def __init__(self, args, tokenizer, descriptions):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.descriptions = {}
        for r in descriptions:
            self.descriptions[r['relation']] = (
                r['positive_template'].replace('[HEAD]', '[H]').replace('[TAIL]', '[T]'),
                r['negative_template'].replace('[HEAD]', '[H]').replace('[TAIL]', '[T]'),
            )

    @abstractmethod
    def __call__(self, batch):
        pass

class TrainCollator(BaseCollator):
    def __init__(self, args, tokenizer, descriptions):
        super().__init__(args, tokenizer, descriptions)

    def get_negative(self, relations):
        num_negs = min(len(relations), self.args.max_relations)
        return random.sample(relations, num_negs - 1)

    def __call__(self, batch):
        relations = list(set([b['relation'] for b in batch]))

        pos_input_texts = []
        neg_input_texts = []
        labels = []

        for b in batch:
            all_relations = set(self.get_negative(relations) + [b['relation']])
            all_relations = sorted(list(all_relations))
            random.shuffle(all_relations)
            for r in all_relations:
                s = self.descriptions[r][0] + ' ' + self.tokenizer.sep_token + ' ' + ' '.join(b['sentence'])
                pos_input_texts.append(s)
                s = self.descriptions[r][1] + ' ' + self.tokenizer.sep_token + ' ' + ' '.join(b['sentence'])
                neg_input_texts.append(s)
                labels.append(1 if b['relation'] == r else 0)

        tokenized_inputs = self.tokenizer(
            pos_input_texts + neg_input_texts,
            padding='longest',
            truncation=True,
            max_length=self.args.max_length,
            return_tensors='pt'
        )

        h_indices = (tokenized_inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('[H]')).nonzero(as_tuple=True)[1].view(-1, 1)
        t_indices = (tokenized_inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('[T]')).nonzero(as_tuple=True)[1].view(-1, 1)
        h_start_indices = (tokenized_inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('[HEAD]')).nonzero(as_tuple=True)[1].view(-1, 1)
        h_end_indices = (tokenized_inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('[/HEAD]')).nonzero(as_tuple=True)[1].view(-1, 1)
        t_start_indices = (tokenized_inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('[TAIL]')).nonzero(as_tuple=True)[1].view(-1, 1)
        t_end_indices = (tokenized_inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('[/TAIL]')).nonzero(as_tuple=True)[1].view(-1, 1)

        boundaries = torch.cat([h_indices, t_indices, h_start_indices, h_end_indices, t_start_indices, t_end_indices], dim=-1)

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'boundaries': boundaries,
            'labels': labels,
        }

class EvalCollator(BaseCollator):
    def __init__(self, args, tokenizer, descriptions, relations):
        super().__init__(args, tokenizer, descriptions)
        self.relations = relations

    def __call__(self, batch):
        pos_input_texts = []
        neg_input_texts = []

        golden_labels = [self.relations.index(b['relation']) for b in batch]

        for b in batch:
            for r in self.relations:
                s = self.descriptions[r][0] + ' ' + self.tokenizer.sep_token + ' ' + ' '.join(b['sentence'])
                pos_input_texts.append(s)
                s = self.descriptions[r][1] + ' ' + self.tokenizer.sep_token + ' ' + ' '.join(b['sentence'])
                neg_input_texts.append(s)

        tokenized_inputs = self.tokenizer(
            pos_input_texts + neg_input_texts,
            padding='longest',
            truncation=True,
            max_length=self.args.max_length,
            return_tensors='pt'
        )

        h_indices = (tokenized_inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('[H]')).nonzero(as_tuple=True)[1].view(-1, 1)
        t_indices = (tokenized_inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('[T]')).nonzero(as_tuple=True)[1].view(-1, 1)
        h_start_indices = (tokenized_inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('[HEAD]')).nonzero(as_tuple=True)[1].view(-1, 1)
        h_end_indices = (tokenized_inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('[/HEAD]')).nonzero(as_tuple=True)[1].view(-1, 1)
        t_start_indices = (tokenized_inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('[TAIL]')).nonzero(as_tuple=True)[1].view(-1, 1)
        t_end_indices = (tokenized_inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('[/TAIL]')).nonzero(as_tuple=True)[1].view(-1, 1)

        boundaries = torch.cat([h_indices, t_indices, h_start_indices, h_end_indices, t_start_indices, t_end_indices], dim=-1)

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'boundaries': boundaries,
            'golden_labels': golden_labels,
        }