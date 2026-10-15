import os
import torch
import random
import numpy as np
from config import parsers
from training import Trainer
from transformers import set_seed

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

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

if __name__ == '__main__':
    args = parsers()

    for s in range(args.num_seeds):
        args.seed = s
        trainer = Trainer(args)
        trainer.train()