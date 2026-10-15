import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from modeling import RelationClassifier
from utils import load_dataset, compute_macro_PRF
from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS

class Trainer:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
        self.model = RelationClassifier(args).to(args.device)

        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[H]', '[T]', '[HEAD]', '[/HEAD]', '[TAIL]', '[/TAIL]']})
        self.model.encoder.resize_token_embeddings(len(self.tokenizer))

    def training(self, data_loader, optimizer, num_epochs):
        num_steps = num_epochs * len(data_loader)
        num_warmup_steps = int(num_steps * self.args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_steps)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(num_epochs):
            self.model.train()

            pbar = tqdm(range(len(data_loader)))
            for i, x in zip(pbar, data_loader):
                optimizer.zero_grad()

                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(self.args.device)

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss = self.model(**x).loss

                if torch.isnan(loss).any():
                    print('Warning: NaN loss detected!')
                    continue

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    scheduler.step()
                del x
                torch.cuda.empty_cache()
                description = f"epoch: {epoch} | step: {i} | loss: {loss.item():.4f}"
                pbar.set_description(description)

            self.evaluate(epoch, 'test')

    def train(self):
        train_loader = load_dataset(self.args, self.tokenizer, load_type='train')

        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if 'bias' not in name]
        encoder_parameters = [name for name, _ in self.model.named_parameters() if 'encoder' in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and n not in encoder_parameters and p.requires_grad)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.others_lr
            },
            {
                "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and n not in encoder_parameters and p.requires_grad)],
                "weight_decay": 0.0,
                "lr": self.args.others_lr
            },
            {
                "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and n in encoder_parameters and p.requires_grad)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.encoder_lr
            },
            {
                "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and n in encoder_parameters and p.requires_grad)],
                "weight_decay": 0.0,
                "lr": self.args.encoder_lr
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        self.training(train_loader, optimizer, self.args.num_epochs)

    def evaluate(self, epoch, eval_type):
        eval_loader = load_dataset(self.args, self.tokenizer, load_type=eval_type)

        self.model.eval()

        all_preds = []
        all_golds = []

        pbar = tqdm(range(len(eval_loader)))
        for i, x in zip(pbar, eval_loader):
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(self.args.device)

            with torch.no_grad():
                logits = self.model(**x).logits.view(len(x['golden_labels']), -1)

            preds = torch.argmax(logits, dim=-1).tolist()

            all_preds.extend(preds)
            all_golds.extend(x['golden_labels'])

        all_preds = np.array(all_preds)
        all_golds = np.array(all_golds)

        precision, recall, f1 = compute_macro_PRF(all_preds, all_golds)

        print('seed: ', self.args.seed)
        print('epoch: ', epoch)
        print('precision: ', precision)
        print('recall: ', recall)
        print('f1: ', f1)