import numpy as np
import torch
import time

from sklearn.metrics._ranking import roc_auc_score, average_precision_score
from tqdm import tqdm
from modules.model import MCGAD
from modules.train import train_model, eval_model

def run_experiment(args, seed, device, dataloader, ano_label, edge_index):
    # Create MCGAD model
    model = MCGAD(
        g=dataloader.g,
        feature=dataloader.en,
        edge_index=edge_index,
        n_in=dataloader.en.shape[1],
        n_hidden=args.n_hidden,
        bn = args.bn,
        beta = args.beta
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    loss_function = torch.nn.BCELoss()

    print(f"Seed {seed}")
    torch.cuda.reset_peak_memory_stats()
    state_path, stats, time_train = train_model(
        args, dataloader, model, optimizer, loss_function
    )
    mem_train = torch.cuda.max_memory_allocated()
    model.load_state_dict(torch.load(state_path))
    torch.cuda.reset_peak_memory_stats()
    score, time_test = eval_model(args, dataloader, model, ano_label)
    mem_test = torch.cuda.max_memory_allocated()

    auc = roc_auc_score(ano_label, score)
    ap = average_precision_score(ano_label, score)

    stats["mem_train"] = mem_train
    stats["mem_test"] = mem_test
    stats["time_train"] = time_train
    stats["time_test"] = time_test
    stats["time_all"] = time_test + time_train
    stats["AUC"] = auc
    stats["AP"] = ap   
    return model, stats
