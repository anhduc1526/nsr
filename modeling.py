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

        self.bridge_mlp = nn.Sequential(
            nn.Linear(4 * args.hidden_size, args.hidden_size),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, 4 * args.hidden_size),
        )
        nn.init.zeros_(self.bridge_mlp[-1].weight)
        nn.init.zeros_(self.bridge_mlp[-1].bias)

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

    def _reason_step(self, v, batch_size, total_size, use_duality, ablation, is_eval, speak_w=None):
        """Run one contrapositive, graph-order-scaled neuro-symbolic reasoning step."""
        z = torch.matmul(self.proj1(v), self.proj2(v).transpose(-2, -1))
        if use_duality:
            imply_logits = 0.5 * (z + z.transpose(-2, -1).roll(shifts=(batch_size, batch_size), dims=(0, 1)))
        else:
            imply_logits = z

        if is_eval:
            weights = imply_logits.clone().fill_diagonal_(torch.finfo(imply_logits.dtype).min).sigmoid_()
        else:
            weights = imply_logits.clone().fill_diagonal_(torch.finfo(imply_logits.dtype).min)
            weights = torch.sigmoid(weights)

        src = v if speak_w is None else (v * speak_w)
        graph_scale = 255.0 / max(1, total_size - 1)
        temp = torch.matmul(weights, src) * graph_scale
        if ablation == 'wo_local':
            out_v = temp
        else:
            ratio = self.gate(torch.cat([v, temp], dim=-1))
            out_v = ratio * v + (1.0 - ratio) * temp
        return out_v, imply_logits

    def _compute_hop_loss(self, step_logits, step_imply_logits, labels, batch_size, use_duality, ablation, imply_labels):
        """Compute prediction and implication loss for a single reasoning hop."""
        if use_duality:
            pos_loss = F.binary_cross_entropy_with_logits(step_logits[:batch_size], labels.float())
            neg_loss = F.binary_cross_entropy_with_logits(step_logits[batch_size:], (1 - labels).float())
            pred_loss = 0.5 * (pos_loss + neg_loss)
        else:
            pred_loss = F.binary_cross_entropy_with_logits(step_logits, labels.float())

        if ablation in ('wo_reasoner', 'wo_implication_loss'):
            return self.args.zeta_1 * pred_loss
        imply_loss = F.binary_cross_entropy_with_logits(step_imply_logits, imply_labels.to(step_imply_logits.dtype))
        return self.args.zeta_1 * pred_loss + self.args.zeta_2 * imply_loss

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

        if ablation != 'wo_reasoner':
            v0 = triplet_embeds
            is_eval = labels is None

            # Hop 1: Initial contrapositive reasoning & certainty estimation
            v1, imp1 = self._reason_step(v0, batch_size, total_size, use_duality, ablation, is_eval)
            logits1 = self.classifier(v1).squeeze(-1)

            # P2 (Hop1-Certainty-Gate): Pair-symmetric certainty weighting (preserves exact 1:1 True/False balance)
            speak_w = None
            if use_duality:
                with torch.no_grad():
                    p1 = torch.sigmoid(logits1.float())
                    cert = 0.25 + 0.75 * (p1[:batch_size] - p1[batch_size:]).abs()
                    cert = cert / cert.mean().clamp(min=1e-6)
                    speak_w = torch.cat([cert, cert], dim=0).to(v0.dtype).unsqueeze(-1)

            # P1 (ZI-2Hop): Normalized Zero-Init Bridge Residual into Hop 2
            u1 = F.layer_norm(v1.float(), (v1.size(-1),)).to(v0.dtype)
            v_bridge = v0 + self.bridge_mlp(u1)
            v2, imp2 = self._reason_step(v_bridge, batch_size, total_size, use_duality, ablation, is_eval, speak_w=speak_w)
            logits = self.classifier(v2).squeeze(-1)
        else:
            logits1, imp1, imp2 = None, None, None
            logits = self.classifier(triplet_embeds).squeeze(-1)

        if use_duality:
            pos_logits = logits[:batch_size]
            neg_logits = logits[batch_size:]
        else:
            pos_logits = logits
            neg_logits = None

        loss = None
        if labels is not None:
            imply_labels = None
            if ablation not in ('wo_reasoner', 'wo_implication_loss'):
                imply_labels_vec = torch.cat([labels, 1 - labels], dim=0) if use_duality else labels
                imply_labels = (imply_labels_vec.unsqueeze(1) <= imply_labels_vec.unsqueeze(0))

            if ablation != 'wo_reasoner':
                loss1 = self._compute_hop_loss(logits1, imp1, labels, batch_size, use_duality, ablation, imply_labels)
                loss2 = self._compute_hop_loss(logits, imp2, labels, batch_size, use_duality, ablation, imply_labels)
                loss = (1.0 / 3.0) * loss1 + (2.0 / 3.0) * loss2
            else:
                loss = self._compute_hop_loss(logits, None, labels, batch_size, use_duality, ablation, None)

        logits = (pos_logits - neg_logits) if use_duality else pos_logits

        return Output(
            loss=loss,
            logits=logits,
        )