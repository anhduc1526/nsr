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

class RelationClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(args.encoder_name)

        self.header = Entity(args.hidden_size, args.dropout)
        self.tailer = Entity(args.hidden_size, args.dropout)
        self.relationer = Relation(2 * args.hidden_size, args.dropout)

        self.pos_signer = Sign(2 * args.hidden_size, args.dropout)
        self.neg_signer = Sign(2 * args.hidden_size, args.dropout)

        self.proj1 = nn.Linear(4 * args.hidden_size, 4 * args.hidden_size)
        self.proj2 = nn.Linear(4 * args.hidden_size, 4 * args.hidden_size)

        self.gate = nn.Sequential(
            FeedForward(2 * 4 * args.hidden_size, args.dropout),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(2 * 4 * args.hidden_size, 1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Linear(4 * args.hidden_size, 1)

    def forward(self, input_ids, attention_mask, boundaries, labels=None, **kwargs):
        token_embeds = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        batch_size = token_embeds.shape[0] // 2
        device = token_embeds.device

        batch_idx = torch.arange(batch_size * 2, dtype=torch.long, device=device).unsqueeze(1)
        boundary_embeds = token_embeds[batch_idx, boundaries]

        relation_embeds = boundary_embeds[:, :2].view(batch_size * 2, -1)
        relation_embeds = self.relationer(relation_embeds)

        head_embeds = self.header(boundary_embeds[:, 2], boundary_embeds[:, 3])
        tail_embeds = self.tailer(boundary_embeds[:, 4], boundary_embeds[:, 5])
        entity_embeds = torch.cat([head_embeds, tail_embeds], dim=-1)

        pos_entity_embeds = self.pos_signer(entity_embeds[:batch_size])
        neg_entity_embeds = self.neg_signer(entity_embeds[batch_size:])

        pos_relation_embeds = self.pos_signer(relation_embeds[:batch_size])
        neg_relation_embeds = self.neg_signer(relation_embeds[batch_size:])

        pos_triplet_embeds = torch.cat([pos_relation_embeds, pos_entity_embeds], dim=-1)
        neg_triplet_embeds = torch.cat([neg_relation_embeds, neg_entity_embeds], dim=-1)

        triplet_embeds = torch.cat([pos_triplet_embeds, neg_triplet_embeds], dim=0)

        imply_logits = torch.matmul(self.proj1(triplet_embeds), self.proj2(triplet_embeds).transpose(-2, -1))

        weights = imply_logits.clone().fill_diagonal_(torch.finfo(imply_logits.dtype).min)
        weights = F.sigmoid(weights)

        temp = torch.matmul(weights, triplet_embeds)
        ratio = self.gate(torch.cat([triplet_embeds, temp], dim=-1))

        triplet_embeds = ratio * triplet_embeds + (1.0 - ratio) * temp

        logits = self.classifier(triplet_embeds).squeeze(-1)

        pos_logits = logits[:batch_size]
        neg_logits = logits[batch_size:]

        loss = None
        if labels is not None:
            pos_labels = labels
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels.float())

            neg_labels = 1 - labels
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels.float())

            pred_loss = (pos_loss + neg_loss) / 2.0

            imply_labels = torch.cat([pos_labels, neg_labels], dim=0)
            imply_labels = (1 - imply_labels.unsqueeze(1) + imply_labels.unsqueeze(0)).clamp(max=1)
            imply_loss = F.binary_cross_entropy_with_logits(imply_logits, imply_labels.float())

            loss = self.args.zeta_1 * pred_loss + self.args.zeta_2 * imply_loss

        logits = pos_logits - neg_logits

        return Output(
            loss=loss,
            logits=logits,
        )