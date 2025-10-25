import time
from sklearn.metrics import roc_auc_score
import json
import os
import copy
import argparse
import numpy as np
import torch

from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.model import Dataloader, load_or_precompute_two_hop
from modules.utils import load_dataset, set_random_seeds, rescale
from modules.experiment import run_experiment

parser = argparse.ArgumentParser(description='MCGAD')
parser.add_argument('--dataset', type=str, default='books')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--beta', type=float, default=0.5)

parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--k', type=int, default=1)

parser.add_argument('--resultdir', type=str, default='results')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--bn', type=bool, default=False)

parser.add_argument('--batch_size', type=int, default=-1)
if __name__ == '__main__':
    args = parser.parse_args()

    seed_list = [6, 25, 248, 1115, 11224]      
    auc_list = []
    ap_list = []
    time_train_list, time_test_list, time_all_list = [], [], []
    mem_train_list, mem_test_list = [], []

    for seed in seed_list:
        print(f"\n=== Running experiment with seed={seed} ===")
        set_random_seeds(seed)

        # Setup torch
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # Load dataset
        g, features, ano_label, _, _, edge_index = load_dataset(args.dataset)
        print(ano_label)
        features = torch.FloatTensor(features)
        if args.batch_size == -1:
            features = features.to(device)
        g = g.to(device)
        dataloader = Dataloader(g, features, args.k, dataset_name=args.dataset)

        if not os.path.isdir("./ckpt"):
            os.makedirs("./ckpt")

        # Run the experiment
        model, stats = run_experiment(args, seed, device, dataloader, ano_label, edge_index)
        print(stats["AUC"], stats["AP"])

        auc_list.append(stats["AUC"])
        ap_list.append(stats["AP"])
        time_train_list.append(stats["time_train"])
        time_all_list.append(stats["time_all"])
        mem_train_list.append(stats["mem_train"])
        time_test_list.append(stats["time_test"])
        mem_test_list.append(stats["mem_test"])


    mean_auc = sum(auc_list) / len(auc_list)
    mean_ap = sum(ap_list) / len(ap_list)
    mean_time_train = sum(time_train_list) / len(time_train_list)
    mean_mem_train = sum(mem_train_list) / len(mem_train_list)
    mean_time_test = sum(time_test_list) / len(time_test_list)
    mean_mem_test = sum(mem_test_list) / len(mem_test_list)
    mean_time_all = sum(time_all_list) / len(time_all_list)

    std_auc = np.std(auc_list)
    std_ap = np.std(ap_list)
    std_time_train = np.std(time_train_list)
    std_mem_train = np.std(mem_train_list)
    std_time_test = np.std(time_test_list)
    std_mem_test = np.std(mem_test_list)
    std_time_all = np.std(mean_time_all)
    
    print("\n=== Average over seeds ===")
    print(f"AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"AP: {mean_ap:.4f} ± {std_ap:.4f}")
    print(f"Time (Train): {mean_time_train:.4f} ± {std_time_train:.4f}s")
    print(f"Mem (Train): {mean_mem_train/1024/1024:.4f} ± {std_mem_train/1024/1024:.4f} MB")
    print(f"Time (Test): {mean_time_test:.4f} ± {std_time_test:.4f}s")
    print(f"Mem (Test): {mean_mem_test/1024/1024:.4f} ± {std_mem_test/1024/1024:.4f} MB")
    print(f"Time (all): {mean_time_all:.4f} ± {mean_time_all:.4f}s")


    log_file = getattr(args, "log_file", f"results.log")
    with open(log_file, "a") as f:
        f.write(f"\n=== Average over seeds [{args.alpha, args.beta, args.lr, args.weight_decay, args.bn, args.num_epoch}] - {args.dataset} ===\n")
        f.write(f"AUC: {mean_auc:.4f} ± {std_auc:.4f}\n")
        f.write(f"AP: {mean_ap:.4f} ± {std_ap:.4f}\n")
        f.write(f"Time (Train): {mean_time_train:.4f} ± {std_time_train:.4f}s\n")
        f.write(f"Mem (Train): {mean_mem_train/1024/1024:.4f} ± {std_mem_train/1024/1024:.4f} MB\n")
        f.write(f"Time (Test): {mean_time_test:.4f} ± {std_time_test:.4f}s\n")
        f.write(f"Mem (Test): {mean_mem_test/1024/1024:.4f} ± {std_mem_test/1024/1024:.4f} MB\n")
        f.write(f"Time (all): {mean_time_all:.4f} ± {mean_time_all:.4f}s\n")
        f.write("=" * 40 + "\n")