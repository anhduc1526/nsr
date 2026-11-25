import os
import re
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
        dataset = WikiZSLDataset(args, data, tokenizer)
    elif args.dataset == 'fewrel':
        dataset = FewRelDataset(args, data, tokenizer)

    pin_memory = torch.cuda.is_available()
    if load_type == 'train':
        train_collator = TrainCollator(args, tokenizer, descriptions)
        data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=train_collator, num_workers=args.num_workers, pin_memory=pin_memory)
    else:
        with open('./Datasets/' + args.dataset + '/' + args.num_unseen_labels + '/seed_' + str(args.seed) + '/' + load_type + '_labels.json', 'r', encoding='utf-8') as f:
            relations = json.load(f)

        eval_collator = EvalCollator(args, tokenizer, descriptions, relations)
        data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=True, collate_fn=eval_collator, num_workers=args.num_workers, pin_memory=pin_memory)

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
    def __init__(self, args, data, tokenizer=None):
        super().__init__()
        self.args = args
        self.data = data
        self.encoded_sents = None
        if tokenizer is not None:
            hs_id = tokenizer.convert_tokens_to_ids('[HEAD]')
            he_id = tokenizer.convert_tokens_to_ids('[/HEAD]')
            ts_id = tokenizer.convert_tokens_to_ids('[TAIL]')
            te_id = tokenizer.convert_tokens_to_ids('[/TAIL]')
            encoded = []
            chunk_size = 4096
            for start in range(0, len(data), chunk_size):
                chunk = data[start:start + chunk_size]
                texts = [
                    ' '.join(add_entity_markers(d['sentence'], d['head'], d['tail']))
                    for d in chunk
                ]
                batch_ids = tokenizer(texts, add_special_tokens=False)['input_ids']
                for ids in batch_ids:
                    encoded.append((
                        ids,
                        ids.index(hs_id),
                        ids.index(he_id),
                        ids.index(ts_id),
                        ids.index(te_id),
                    ))
            self.encoded_sents = encoded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_i = self.data[index]
        if self.encoded_sents is not None:
            return {
                'encoded_sent': self.encoded_sents[index],
                'relation': data_i['relation'],
                'sample_idx': index,
            }
        return {
            'sentence': add_entity_markers(data_i['sentence'], data_i['head'], data_i['tail']),
            'relation': data_i['relation'],
            'sample_idx': index,
        }

class WikiZSLDataset(BaseDataset):
    def __init__(self, args, data, tokenizer=None):
        super().__init__(args, data, tokenizer)

class FewRelDataset(BaseDataset):
    def __init__(self, args, data, tokenizer=None):
        super().__init__(args, data, tokenizer)

def strip_type_constraints(template):
    """Remove parenthesized entity type constraints from a relation schema template."""
    cleaned = re.sub(r'\s*\([^)]*\)', '', template)
    return re.sub(r'\s+', ' ', cleaned).strip()

class BaseCollator(ABC):
    def __init__(self, args, tokenizer, descriptions):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        ablation = getattr(args, 'ablation', 'none')
        self.use_duality = ablation not in ('wo_duality', 'wo_both')
        remove_types = ablation in ('wo_type_constraints', 'wo_both')
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        h_id = tokenizer.convert_tokens_to_ids('[H]')
        t_id = tokenizer.convert_tokens_to_ids('[T]')
        self.descriptions = {}
        self.pos_schemas = {}
        self.neg_schemas = {}
        for r in descriptions:
            pos_tpl = r['positive_template']
            neg_tpl = r['negative_template']
            if remove_types:
                pos_tpl = strip_type_constraints(pos_tpl)
                neg_tpl = strip_type_constraints(neg_tpl)
            pos_tpl = pos_tpl.replace('[HEAD]', '[H]').replace('[TAIL]', '[T]')
            neg_tpl = neg_tpl.replace('[HEAD]', '[H]').replace('[TAIL]', '[T]')
            self.descriptions[r['relation']] = (pos_tpl, neg_tpl)
            pos_ids = tokenizer.encode(pos_tpl, add_special_tokens=False)
            neg_ids = tokenizer.encode(neg_tpl, add_special_tokens=False)
            self.pos_schemas[r['relation']] = (pos_ids, pos_ids.index(h_id), pos_ids.index(t_id))
            self.neg_schemas[r['relation']] = (neg_ids, neg_ids.index(h_id), neg_ids.index(t_id))

    def _build_pair(self, sch_entry, sent_entry):
        sch_ids, h_pos, t_pos = sch_entry
        sent_ids, hs, he, ts, te = sent_entry
        off = len(sch_ids) + 2
        seq = [self.cls_id] + sch_ids + [self.sep_id] + sent_ids + [self.sep_id]
        if len(seq) > self.args.max_length:
            seq = seq[:self.args.max_length - 1] + [self.sep_id]
        max_idx = len(seq) - 1
        bound = (
            min(1 + h_pos, max_idx),
            min(1 + t_pos, max_idx),
            min(off + hs, max_idx),
            min(off + he, max_idx),
            min(off + ts, max_idx),
            min(off + te, max_idx),
        )
        return seq, bound

    def _pack_pairs(self, pos_seqs, pos_bounds, neg_seqs, neg_bounds):
        seqs = pos_seqs + neg_seqs if self.use_duality else pos_seqs
        bounds = pos_bounds + neg_bounds if self.use_duality else pos_bounds
        max_len = max(len(s) for s in seqs)
        pad_id = self.pad_id
        padded = [s + [pad_id] * (max_len - len(s)) if len(s) < max_len else s for s in seqs]
        input_ids = torch.tensor(padded, dtype=torch.long)
        lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        attention_mask = (torch.arange(max_len, dtype=torch.long).unsqueeze(0) < lens.unsqueeze(1)).long()
        boundaries = torch.tensor(bounds, dtype=torch.long)
        return input_ids, attention_mask, boundaries

    @abstractmethod
    def __call__(self, batch):
        pass

class TrainCollator(BaseCollator):
    def __init__(self, args, tokenizer, descriptions):
        super().__init__(args, tokenizer, descriptions)

    def get_negative(self, relations, gold_rel):
        neg_pool = [r for r in relations if r != gold_rel]
        k = min(len(neg_pool), self.args.max_relations - 1)
        return random.sample(neg_pool, k)

    def __call__(self, batch):
        relations = list(set([b['relation'] for b in batch]))

        pos_seqs, pos_bounds = [], []
        neg_seqs, neg_bounds = [], []
        labels = []

        for b in batch:
            sent_entry = b['encoded_sent']
            all_relations = self.get_negative(relations, b['relation']) + [b['relation']]
            random.shuffle(all_relations)
            for r in all_relations:
                seq, bound = self._build_pair(self.pos_schemas[r], sent_entry)
                pos_seqs.append(seq)
                pos_bounds.append(bound)
                if self.use_duality:
                    n_seq, n_bound = self._build_pair(self.neg_schemas[r], sent_entry)
                    neg_seqs.append(n_seq)
                    neg_bounds.append(n_bound)
                labels.append(1 if b['relation'] == r else 0)

        input_ids, attention_mask, boundaries = self._pack_pairs(pos_seqs, pos_bounds, neg_seqs, neg_bounds)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'boundaries': boundaries,
            'labels': labels,
        }

class EvalCollator(BaseCollator):
    def __init__(self, args, tokenizer, descriptions, relations):
        super().__init__(args, tokenizer, descriptions)
        self.relations = relations

    def __call__(self, batch):
        pos_seqs, pos_bounds = [], []
        neg_seqs, neg_bounds = [], []

        golden_labels = [self.relations.index(b['relation']) for b in batch]

        for b in batch:
            sent_entry = b['encoded_sent']
            for r in self.relations:
                seq, bound = self._build_pair(self.pos_schemas[r], sent_entry)
                pos_seqs.append(seq)
                pos_bounds.append(bound)
                if self.use_duality:
                    n_seq, n_bound = self._build_pair(self.neg_schemas[r], sent_entry)
                    neg_seqs.append(n_seq)
                    neg_bounds.append(n_bound)

        input_ids, attention_mask, boundaries = self._pack_pairs(pos_seqs, pos_bounds, neg_seqs, neg_bounds)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'boundaries': boundaries,
            'golden_labels': golden_labels,
            'sample_indices': [b['sample_idx'] for b in batch],
        }