import faulthandler
import gc
import glob
import json
import os
import sys
from collections import Counter, defaultdict

faulthandler.enable()


def _evict_torch_so_page_cache():
    """Evict cached RAM pages of torch shared libraries so fresh copies are read from disk."""
    if not hasattr(os, 'posix_fadvise') or not hasattr(os, 'POSIX_FADV_DONTNEED'):
        return
    for sp in sys.path:
        torch_lib = os.path.join(sp, 'torch', 'lib')
        if os.path.isdir(torch_lib):
            for so_path in glob.glob(os.path.join(torch_lib, '*.so*')):
                try:
                    fd = os.open(so_path, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                except OSError:
                    pass


_evict_torch_so_page_cache()
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import random
import numpy as np
from config import parsers
from training import Trainer
from transformers import set_seed
from utils import add_entity_markers

os.environ['TOKENIZERS_PARALLELISM'] = '0'

seed = 42
set_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)


def _format_confused_pairs(pair_counter, pair_seeds=None):
    """Sort confused (gold, pred) relation pairs by frequency descending."""
    sorted_pairs = sorted(pair_counter.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    results = []
    for (gold_rel, pred_rel), count in sorted_pairs:
        entry = {
            'gold_relation': gold_rel,
            'pred_relation': pred_rel,
            'count': count,
        }
        if pair_seeds is not None:
            entry['seeds'] = sorted(pair_seeds[(gold_rel, pred_rel)])
        results.append(entry)
    return results


def _mean_metrics(metric_dicts):
    """Compute arithmetic mean of precision, recall, and f1 across seeds."""
    if not metric_dicts:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    n = len(metric_dicts)
    return {
        key: sum(m[key] for m in metric_dicts) / n
        for key in ('precision', 'recall', 'f1')
    }


def build_seed_analysis(args, current_seed, epoch_results):
    """Collect per-seed summary metrics and samples misclassified in every epoch."""
    best_ep_res = max(epoch_results, key=lambda r: r['f1'])
    last_ep_res = epoch_results[-1]
    raw_data = best_ep_res['raw_data']

    rel_totals = Counter()
    rel_wrong_all = Counter()
    rel_wrong_best = Counter()
    confused_all_counter = Counter()
    confused_best_counter = Counter()
    error_samples = []
    wrong_best_count = 0

    for sample_idx, item in enumerate(raw_data):
        gold_rel = best_ep_res['golds_by_idx'][sample_idx]
        preds_by_epoch = {str(r['epoch']): r['preds_by_idx'][sample_idx] for r in epoch_results}
        pred_best = best_ep_res['preds_by_idx'][sample_idx]

        rel_totals[gold_rel] += 1
        wrong_all = all(r['preds_by_idx'][sample_idx] != gold_rel for r in epoch_results)
        wrong_best = pred_best != gold_rel

        if wrong_best:
            wrong_best_count += 1
            rel_wrong_best[gold_rel] += 1
            confused_best_counter[(gold_rel, pred_best)] += 1

        if wrong_all:
            rel_wrong_all[gold_rel] += 1
            confused_all_counter[(gold_rel, pred_best)] += 1
            head_entity = ' '.join(item['sentence'][item['head'][0]:item['head'][1]])
            tail_entity = ' '.join(item['sentence'][item['tail'][0]:item['tail'][1]])
            error_samples.append({
                'record_type': 'seed_error_sample',
                'dataset': args.dataset,
                'num_unseen_labels': args.num_unseen_labels,
                'ablation': getattr(args, 'ablation', 'none'),
                'seed': current_seed,
                'sample_idx': sample_idx,
                'head_entity': head_entity,
                'tail_entity': tail_entity,
                'sentence': ' '.join(item['sentence']),
                'marked_sentence': ' '.join(add_entity_markers(item['sentence'], item['head'], item['tail'])),
                'gold_relation': gold_rel,
                'predictions_by_epoch': preds_by_epoch,
                'pred_at_best_epoch': pred_best,
                'consistent_wrong_prediction': len(set(preds_by_epoch.values())) == 1,
            })

    per_relation_stats = {}
    for rel in best_ep_res['relations']:
        total = rel_totals[rel]
        w_all = rel_wrong_all[rel]
        w_best = rel_wrong_best[rel]
        per_relation_stats[rel] = {
            'total_samples': total,
            'wrong_all_epochs_count': w_all,
            'wrong_all_epochs_rate': (w_all / total) if total else 0.0,
            'wrong_best_epoch_count': w_best,
            'wrong_best_epoch_rate': (w_best / total) if total else 0.0,
        }

    seed_summary = {
        'record_type': 'seed_summary',
        'dataset': args.dataset,
        'num_unseen_labels': args.num_unseen_labels,
        'ablation': getattr(args, 'ablation', 'none'),
        'seed': current_seed,
        'relations': best_ep_res['relations'],
        'total_samples': len(raw_data),
        'num_wrong_all_epochs': len(error_samples),
        'num_wrong_best_epoch': wrong_best_count,
        'best_epoch': best_ep_res['epoch'],
        'best_epoch_metrics': {
            'precision': best_ep_res['precision'],
            'recall': best_ep_res['recall'],
            'f1': best_ep_res['f1'],
        },
        'last_epoch': last_ep_res['epoch'],
        'last_epoch_metrics': {
            'precision': last_ep_res['precision'],
            'recall': last_ep_res['recall'],
            'f1': last_ep_res['f1'],
        },
        'epoch_metrics': [
            {
                'epoch': r['epoch'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1'],
            }
            for r in epoch_results
        ],
        'confused_pairs_all_epochs': _format_confused_pairs(confused_all_counter),
        'confused_pairs_best_epoch': _format_confused_pairs(confused_best_counter),
        'per_relation_stats': per_relation_stats,
    }

    return {
        'summary': seed_summary,
        'error_samples': error_samples,
        'confused_all_counter': confused_all_counter,
        'confused_best_counter': confused_best_counter,
    }


def build_m_summary(args, all_seed_analyses):
    """Aggregate metrics, relation error rates, confused pairs, and hard errors across seeds."""
    seed_summaries = [item['summary'] for item in all_seed_analyses]
    mean_best = _mean_metrics([s['best_epoch_metrics'] for s in seed_summaries])
    mean_last = _mean_metrics([s['last_epoch_metrics'] for s in seed_summaries])

    rel_agg = defaultdict(lambda: {'total_samples': 0, 'wrong_all_epochs_count': 0, 'wrong_best_epoch_count': 0, 'confused_with': Counter()})
    confused_all_total = Counter()
    confused_all_seeds = defaultdict(set)
    confused_best_total = Counter()
    confused_best_seeds = defaultdict(set)
    all_hard_error_samples = []
    by_sentence_key = defaultdict(list)

    for analysis in all_seed_analyses:
        summary = analysis['summary']
        s_id = summary['seed']
        for rel, stats in summary['per_relation_stats'].items():
            rel_agg[rel]['total_samples'] += stats['total_samples']
            rel_agg[rel]['wrong_all_epochs_count'] += stats['wrong_all_epochs_count']
            rel_agg[rel]['wrong_best_epoch_count'] += stats['wrong_best_epoch_count']

        for (gold_rel, pred_rel), count in analysis['confused_all_counter'].items():
            confused_all_total[(gold_rel, pred_rel)] += count
            confused_all_seeds[(gold_rel, pred_rel)].add(s_id)
            rel_agg[gold_rel]['confused_with'][pred_rel] += count

        for (gold_rel, pred_rel), count in analysis['confused_best_counter'].items():
            confused_best_total[(gold_rel, pred_rel)] += count
            confused_best_seeds[(gold_rel, pred_rel)].add(s_id)

        for err in analysis['error_samples']:
            all_hard_error_samples.append({
                'seed': err['seed'],
                'sample_idx': err['sample_idx'],
                'gold_relation': err['gold_relation'],
                'pred_at_best_epoch': err['pred_at_best_epoch'],
                'consistent_wrong_prediction': err['consistent_wrong_prediction'],
                'head_entity': err['head_entity'],
                'tail_entity': err['tail_entity'],
            })
            key = (err['sentence'], err['head_entity'], err['tail_entity'], err['gold_relation'])
            by_sentence_key[key].append({
                'seed': err['seed'],
                'sample_idx': err['sample_idx'],
                'pred_at_best_epoch': err['pred_at_best_epoch'],
                'predictions_by_epoch': err['predictions_by_epoch'],
            })

    per_relation_stats = {}
    for rel in sorted(rel_agg.keys()):
        data = rel_agg[rel]
        total = data['total_samples']
        w_all = data['wrong_all_epochs_count']
        w_best = data['wrong_best_epoch_count']
        top_confused = [
            {'pred_relation': pred_rel, 'count': cnt}
            for pred_rel, cnt in sorted(data['confused_with'].items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        per_relation_stats[rel] = {
            'total_samples': total,
            'wrong_all_epochs_count': w_all,
            'wrong_all_epochs_rate': (w_all / total) if total else 0.0,
            'wrong_best_epoch_count': w_best,
            'wrong_best_epoch_rate': (w_best / total) if total else 0.0,
            'top_confused_with': top_confused,
        }

    repeated_hard_errors = []
    for (sentence, head_entity, tail_entity, gold_rel), occurrences in by_sentence_key.items():
        distinct_seeds = {occ['seed'] for occ in occurrences}
        if len(distinct_seeds) >= 2:
            repeated_hard_errors.append({
                'sentence': sentence,
                'head_entity': head_entity,
                'tail_entity': tail_entity,
                'gold_relation': gold_rel,
                'num_seeds': len(distinct_seeds),
                'occurrences': occurrences,
            })

    seed_overview = [
        {
            'seed': s['seed'],
            'best_epoch': s['best_epoch'],
            'best_epoch_metrics': s['best_epoch_metrics'],
            'last_epoch_metrics': s['last_epoch_metrics'],
            'total_samples': s['total_samples'],
            'num_wrong_all_epochs': s['num_wrong_all_epochs'],
            'num_wrong_best_epoch': s['num_wrong_best_epoch'],
        }
        for s in seed_summaries
    ]

    return {
        'record_type': 'm_summary',
        'dataset': args.dataset,
        'num_unseen_labels': args.num_unseen_labels,
        'ablation': getattr(args, 'ablation', 'none'),
        'num_seeds': len(seed_summaries),
        'mean_best_epoch_metrics': mean_best,
        'mean_last_epoch_metrics': mean_last,
        'total_hard_error_samples': len(all_hard_error_samples),
        'seed_overview': seed_overview,
        'per_relation_stats': per_relation_stats,
        'confused_relation_pairs_all_epochs': _format_confused_pairs(confused_all_total, confused_all_seeds),
        'confused_relation_pairs_best_epoch': _format_confused_pairs(confused_best_total, confused_best_seeds),
        'all_hard_error_samples': all_hard_error_samples,
        'repeated_hard_errors_across_seeds': repeated_hard_errors,
    }


def _reconstruct_pair_counter(pair_list):
    """Rebuild Counter((gold_rel, pred_rel) -> count) from serialized summary list."""
    counter = Counter()
    for item in pair_list:
        counter[(item['gold_relation'], item['pred_relation'])] += item['count']
    return counter


def load_existing_seed_analyses(output_path):
    """Load validated completed seeds from an existing error-analysis JSONL file."""
    if not os.path.exists(output_path):
        return {}

    summaries_by_seed = {}
    errors_by_seed = defaultdict(list)

    with open(output_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rt = record.get('record_type')
            if rt == 'seed_summary':
                s = record['seed']
                summaries_by_seed[s] = record
                errors_by_seed[s] = []
            elif rt == 'seed_error_sample':
                errors_by_seed[record['seed']].append(record)

    completed = {}
    for s, summary in sorted(summaries_by_seed.items()):
        err_samples = errors_by_seed[s]
        if len(err_samples) != summary.get('num_wrong_all_epochs', -1):
            continue
        completed[s] = {
            'summary': summary,
            'error_samples': err_samples,
            'confused_all_counter': _reconstruct_pair_counter(summary.get('confused_pairs_all_epochs', [])),
            'confused_best_counter': _reconstruct_pair_counter(summary.get('confused_pairs_best_epoch', [])),
        }
    return completed


def _write_seed_records(f, seed_analysis):
    """Write a seed_summary record and its seed_error_sample records to JSONL."""
    f.write(json.dumps(seed_analysis['summary'], ensure_ascii=False) + '\n')
    for err in seed_analysis['error_samples']:
        f.write(json.dumps(err, ensure_ascii=False) + '\n')


def resolve_error_output_path(args):
    """Resolve JSONL error-analysis path matching logs/gnsr_<dataset>_m<unseen_m>.jsonl."""
    if args.error_output:
        return args.error_output
    m_tag = args.num_unseen_labels.replace('unseen_', '')
    ablation = getattr(args, 'ablation', 'none')
    suffix = '' if ablation == 'none' else f'_{ablation}'
    return f"./logs/gnsr_{args.dataset}_m{m_tag}{suffix}.jsonl"


if __name__ == '__main__':
    args = parsers()
    output_path = resolve_error_output_path(args)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if getattr(args, 'resume', False) and os.path.exists(output_path):
        completed_seeds = load_existing_seed_analyses(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for s in sorted(completed_seeds):
                _write_seed_records(f, completed_seeds[s])
            f.flush()
    else:
        completed_seeds = {}
        with open(output_path, 'w', encoding='utf-8'):
            pass

    all_seed_analyses = []
    for s in range(args.num_seeds):
        if s in completed_seeds:
            print(f'Skipping completed seed: {s} (resumed from {output_path})')
            all_seed_analyses.append(completed_seeds[s])
            continue

        args.seed = s
        trainer = Trainer(args)
        epoch_results = trainer.train()

        seed_analysis = build_seed_analysis(args, s, epoch_results)
        all_seed_analyses.append(seed_analysis)

        with open(output_path, 'a', encoding='utf-8') as f:
            _write_seed_records(f, seed_analysis)
            f.flush()

        del trainer
        del epoch_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _evict_torch_so_page_cache()

    m_summary = build_m_summary(args, all_seed_analyses)
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(m_summary, ensure_ascii=False) + '\n')
        f.flush()