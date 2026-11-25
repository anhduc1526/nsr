import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None

class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x):
        output = self.layers(x)
        return output

class Entity(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.fusion_layer = nn.Linear(4 * hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, start_embeds, end_embeds):
        prod = start_embeds * end_embeds
        diff = start_embeds - end_embeds
        concat = torch.cat((start_embeds, end_embeds, prod, diff), dim=-1)

        output = self.fusion_layer(concat)
        output = self.activation(output)
        output = self.dropout(output)

        residual = (start_embeds + end_embeds) / 2.0
        output = self.layer_norm(residual + output)

        return output

class Relation(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.feed_forward = FeedForward(hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, relation_embeds):
        output = self.feed_forward(relation_embeds)
        output = self.layer_norm(relation_embeds + self.dropout(output))
        return output

class Sign(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            FeedForward(hidden_size, dropout),
            FeedForward(hidden_size, dropout),
            FeedForward(hidden_size, dropout),
        ])
        self.activation = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        weights = self.layers[0](x)
        weights = self.activation(weights)

        output = weights * self.layers[1](x)

        residual = output
        output = self.layers[2](output)
        output = self.layer_norm(residual + self.dropout(output))

        return output

def _dp_partition(sorted_lens, budget, alpha=1024.0, beta=0.002):
    """Partition sorted aligned lengths into chunks respecting token budget while minimizing compute."""
    blocks, n, i = [], len(sorted_lens), 0
    while i < n:
        L, j = sorted_lens[i], i
        while j < n and sorted_lens[j] == L:
            j += 1
        max_c = max(1, budget // L)
        for cur in range(i, j, max_c):
            blocks.append((L, cur, min(j, cur + max_c)))
        i = j

    U = len(blocks)
    dp, par = [float('inf')] * (U + 1), [-1] * (U + 1)
    dp[0] = 0.0
    for j in range(1, U + 1):
        Lj, end_pos = blocks[j - 1][0], blocks[j - 1][2]
        for i in range(j - 1, -1, -1):
            cnt = end_pos - blocks[i][1]
            if cnt * Lj > budget and i < j - 1:
                break
            cost = dp[i] + alpha + cnt * (Lj + beta * Lj * Lj)
            if cost < dp[j]:
                dp[j], par[j] = cost, i

    chunks, cur = [], U
    while cur > 0:
        prv = par[cur]
        chunks.append((blocks[prv][1], blocks[cur - 1][2], blocks[cur - 1][0]))
        cur = prv
    chunks.reverse()
    return chunks


class RelationClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(args.encoder_name)
        if hasattr(self.encoder, 'config'):
            self.encoder.config.use_cache = False
        self._unckpt_budget = None

        self.header = Entity(args.hidden_size, args.dropout)
        self.tailer = Entity(args.hidden_size, args.dropout)
        self.relationer = Relation(2 * args.hidden_size, args.dropout)

        self.pos_signer = Sign(2 * args.hidden_size, args.dropout)
        self.neg_signer = Sign(2 * args.hidden_size, args.dropout)

        self.proj1 = nn.Linear(4 * args.hidden_size, 4 * args.hidden_size)
        self.proj2 = nn.Linear(4 * args.hidden_size, 4 * args.hidden_size)

        self.gate = nn.Sequential(
            nn.Linear(2 * 4 * args.hidden_size, args.hidden_size),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, 1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Linear(4 * args.hidden_size, 1)

    def _set_grad_checkpointing(self, enable):
        is_ckpt = getattr(self.encoder, 'is_gradient_checkpointing', False)
        if enable and not is_ckpt:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        elif not enable and is_ckpt:
            self.encoder.gradient_checkpointing_disable()

    def _encode_boundaries(self, input_ids, attention_mask, boundaries):
        device = input_ids.device
        total_size = input_ids.shape[0]
        lengths = attention_mask.sum(dim=1)
        aligned = ((lengths + 7) // 8 * 8).clamp(min=8, max=input_ids.shape[1])
        perm = torch.argsort(aligned)
        budget = 16384 if self.training else 12288
        chunks = _dp_partition(aligned[perm].tolist(), budget=budget)

        if self.training:
            if self._unckpt_budget is None:
                if input_ids.is_cuda:
                    total_gib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
                    self._unckpt_budget = int(max(0.0, total_gib - 6.2) * 1600)
                else:
                    self._unckpt_budget = 10 ** 9
            unckpt_budget = self._unckpt_budget
        else:
            self._set_grad_checkpointing(False)
            unckpt_budget = 10 ** 9

        unckpt_used = 0
        gathered = []
        for s, e, L_max in chunks:
            if self.training:
                chunk_tokens = (e - s) * L_max
                if unckpt_used + chunk_tokens <= unckpt_budget:
                    self._set_grad_checkpointing(False)
                    unckpt_used += chunk_tokens
                else:
                    self._set_grad_checkpointing(True)
            idx = perm[s:e]
            out = self.encoder(
                input_ids=input_ids[idx, :L_max],
                attention_mask=attention_mask[idx, :L_max],
            ).last_hidden_state
            row = torch.arange(e - s, dtype=torch.long, device=device).unsqueeze(1)
            gathered.append(out[row, boundaries[idx]])

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(total_size, dtype=torch.long, device=device)
        return torch.cat(gathered, dim=0)[inv_perm]

    def forward(self, input_ids, attention_mask, boundaries, labels=None, **kwargs):
        ablation = getattr(self.args, 'ablation', 'none')
        use_duality = ablation not in ('wo_duality', 'wo_both')

        boundary_embeds = self._encode_boundaries(input_ids, attention_mask, boundaries)

        if use_duality:
            batch_size = boundary_embeds.shape[0] // 2
            total_size = batch_size * 2
        else:
            batch_size = boundary_embeds.shape[0]
            total_size = batch_size

        relation_embeds = boundary_embeds[:, :2].reshape(total_size, -1)
        relation_embeds = self.relationer(relation_embeds)

        head_embeds = self.header(boundary_embeds[:, 2], boundary_embeds[:, 3])
        tail_embeds = self.tailer(boundary_embeds[:, 4], boundary_embeds[:, 5])
        entity_embeds = torch.cat([head_embeds, tail_embeds], dim=-1)

        if use_duality:
            pos_both = self.pos_signer(torch.cat([relation_embeds[:batch_size], entity_embeds[:batch_size]], dim=0))
            neg_both = self.neg_signer(torch.cat([relation_embeds[batch_size:], entity_embeds[batch_size:]], dim=0))
            pos_triplet_embeds = torch.cat([pos_both[:batch_size], pos_both[batch_size:]], dim=-1)
            neg_triplet_embeds = torch.cat([neg_both[:batch_size], neg_both[batch_size:]], dim=-1)
            triplet_embeds = torch.cat([pos_triplet_embeds, neg_triplet_embeds], dim=0)
        else:
            pos_both = self.pos_signer(torch.cat([relation_embeds, entity_embeds], dim=0))
            triplet_embeds = torch.cat([pos_both[:batch_size], pos_both[batch_size:]], dim=-1)

        imply_logits = None
        if ablation != 'wo_reasoner':
            imply_logits = torch.matmul(self.proj1(triplet_embeds), self.proj2(triplet_embeds).transpose(-2, -1))
            if labels is None:
                weights = imply_logits.fill_diagonal_(torch.finfo(imply_logits.dtype).min).sigmoid_()
            else:
                weights = imply_logits.clone().fill_diagonal_(torch.finfo(imply_logits.dtype).min)
                weights = F.sigmoid(weights)

            temp = torch.matmul(weights, triplet_embeds)
            if ablation == 'wo_local':
                triplet_embeds = temp
            else:
                ratio = self.gate(torch.cat([triplet_embeds, temp], dim=-1))
                triplet_embeds = ratio * triplet_embeds + (1.0 - ratio) * temp

        logits = self.classifier(triplet_embeds).squeeze(-1)

        if use_duality:
            pos_logits = logits[:batch_size]
            neg_logits = logits[batch_size:]
        else:
            pos_logits = logits
            neg_logits = None

        loss = None
        if labels is not None:
            if use_duality:
                pos_labels = labels
                pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels.float())

                neg_labels = 1 - labels
                neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels.float())

                pred_loss = (pos_loss + neg_loss) / 2.0
                imply_labels_vec = torch.cat([pos_labels, neg_labels], dim=0)
            else:
                pred_loss = F.binary_cross_entropy_with_logits(pos_logits, labels.float())
                imply_labels_vec = labels

            if ablation in ('wo_reasoner', 'wo_implication_loss'):
                loss = self.args.zeta_1 * pred_loss
            else:
                imply_labels = (imply_labels_vec.unsqueeze(1) <= imply_labels_vec.unsqueeze(0)).to(imply_logits.dtype)
                imply_loss = F.binary_cross_entropy_with_logits(imply_logits, imply_labels)
                loss = self.args.zeta_1 * pred_loss + self.args.zeta_2 * imply_loss

        logits = (pos_logits - neg_logits) if use_duality else pos_logits

        return Output(
            loss=loss,
            logits=logits,
        )