import time
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from modules.utils import rescale
from tqdm import tqdm
from pathlib import Path

def train_model(args, dataloader, model, optimizer, ano_label=None):
    if ano_label is not None:
        ano_label = torch.as_tensor(ano_label, device=dataloader.en.device)

    stats = {
        "best_loss": 1e9,
        "best_epoch": -1,
    }
    state_path = getattr(args, "checkpoint_path", None) or f'./ckpt/{args.dataset}.pkl'
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    time_train = time.time()
    x_ego, x_2hop = dataloader.get_data()

    raw_sim = F.cosine_similarity(x_ego, x_2hop, dim=1)

    for epoch in tqdm(range(args.num_epoch), desc="Training Epochs"):
        model.train()
        optimizer.zero_grad()
        x_ego, x_2hop = dataloader.get_data()

        score, loss_uni, gad_loss = model(x_ego, x_2hop)
        score = rescale(score)

        raw_score = rescale(raw_sim).detach().unsqueeze(0)
        loss_recon = torch.nn.functional.mse_loss(score, raw_score)
        loss = args.alpha * loss_uni + args.beta * loss_recon + gad_loss
        loss.backward()

        if loss < stats["best_loss"]:
            stats["best_loss"] = loss
            stats["best_epoch"] = epoch
            torch.save(model.state_dict(), state_path)
        optimizer.step()

    time_train = time.time() - time_train
    return state_path, stats, time_train

def eval_model(args, dataloader, model, ano_label):
    model.eval()
    with torch.no_grad():
        time_test = time.time()
        if args.batch_size == -1:
            score = model(dataloader.en, dataloader.eg)
            score = - score[0].cpu().numpy()
        else:
            score = []
            en = dataloader.en
            eg = dataloader.eg
            i = 0
            while i * args.batch_size < len(en):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, len(en))
                en_batch, eg_batch = en[start_index:end_index], eg[start_index:end_index]
                en_batch, eg_batch = [x.to("cuda") for x in [en_batch, eg_batch]]
                score.append(model(en_batch, eg_batch).detach().cpu().numpy())
                i += 1
            score = np.concatenate(score, axis=1)[0]

        time_test = time.time() - time_test
    return score, time_test
