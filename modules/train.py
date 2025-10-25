import time
from sklearn.metrics import roc_auc_score
import json
import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from modules.utils import rescale
from tqdm import tqdm

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path


def train_model(args, dataloader, model, optimizer, loss_function):

    stats = {
        "best_loss": 1e9,
        "best_epoch": -1,
    }
    state_path = f'./ckpt/{args.dataset}.pkl'
    time_train = time.time()

    for epoch in tqdm(range(args.num_epoch), desc="Training Epochs"):
        model.train()
        optimizer.zero_grad()
        x_ego, x_2hop = dataloader.get_data()

        score, loss_uni = model(x_ego, x_2hop)

        score = rescale(score)
        loss_mono = loss_function(score, dataloader.label_ones)

        loss = (1-args.alpha)*loss_mono + args.alpha*loss_uni
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