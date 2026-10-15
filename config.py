import argparse

def parsers():
    parser = argparse.ArgumentParser(description='G-NSR model of argparse')

    parser.add_argument('--encoder_name', type=str, default='google-bert/bert-base-uncased')
    parser.add_argument('--dataset', type=str, default='fewrel')
    parser.add_argument('--num_unseen_labels', type=str, default='unseen_5')
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--encoder_lr', type=float, default=5e-5)
    parser.add_argument('--others_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_relations', type=int, default=4)
    parser.add_argument('--zeta_1', type=float, default=0.5)
    parser.add_argument('--zeta_2', type=float, default=1.0)

    return parser.parse_args()