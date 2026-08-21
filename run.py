import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from modules.experiment import run_experiment
from modules.model import Dataloader, load_or_compute_xtlx, two_hop_map_to_edge_index
from modules.train import eval_model
from modules.utils import load_dataset, set_random_seeds


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


parser = argparse.ArgumentParser(description="MCGAD")
parser.add_argument("--dataset", type=str, default="book")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--gamma", type=float, default=1.0)

parser.add_argument("--n_hidden", type=int, default=128)
parser.add_argument("--k", type=int, default=1)

parser.add_argument("--resultdir", type=str, default="results")
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--seed_list", type=str, default="")
parser.add_argument("--num_epoch", type=int, default=100)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--bn", type=parse_bool, nargs="?", const=True, default=False)
parser.add_argument("--log_file", type=str, default="results.log")

parser.add_argument("--batch_size", type=int, default=-1)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.seed_list:
        seed_list = [
            int(seed.strip())
            for seed in args.seed_list.split(",")
            if seed.strip()
        ]
    else:
        seed_list = [6005, 248, 1, 811, 2616]
    seed_text = ",".join(str(seed) for seed in seed_list)
    single_seed_prefix = f"seed={seed_list[0]} | " if len(seed_list) == 1 else ""

    auc_list = []
    ap_list = []
    time_train_list = []
    time_test_list = []
    mem_train_list = []
    mem_test_list = []

    os.makedirs("./ckpt", exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    g, features, ano_label, _, _, edge_index = load_dataset(args.dataset)
    features = torch.FloatTensor(features)
    if args.batch_size == -1:
        features = features.to(device)
    g = g.to(device)
    dataloader = Dataloader(g, features, args.k, dataset_name=args.dataset)

    with torch.no_grad():
        raw_similarity = F.cosine_similarity(
            dataloader.en,
            dataloader.eg,
            dim=1,
        )
        raw_local_score = 1 - raw_similarity

        raw_center = dataloader.en.mean(dim=0, keepdim=True)
        raw_global_score = 1 - F.cosine_similarity(
            dataloader.en,
            raw_center,
            dim=1,
        )
        raw_feature_score = (
            0.5 * raw_local_score + 0.5 * raw_global_score
        ).detach().cpu().numpy()

    xl2x_path = Path("xl2x") / f"{args.dataset}.pt"
    xl2x = torch.load(xl2x_path, map_location="cpu")
    xl2x = torch.logit(xl2x.clamp(1e-6, 1 - 1e-6))

    for seed in seed_list:
        print(f"\n=== Running experiment with seed={seed} ===")
        set_random_seeds(seed)

        model, stats = run_experiment(
            args,
            seed,
            device,
            dataloader,
            ano_label,
            edge_index,
            xl2x=xl2x,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model.eval()
        score, time_test = eval_model(args, dataloader, model, ano_label)
        seed_auc = roc_auc_score(ano_label, score)
        seed_ap = average_precision_score(ano_label, score)
        mem_test = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

        auc_list.append(seed_auc)
        ap_list.append(seed_ap)
        print(
            f"seed={seed} | "
            f"AUC: {seed_auc:.4f} | "
            f"AP: {seed_ap:.4f}"
        )

        time_train_list.append(stats["time_train"])
        time_test_list.append(time_test)
        mem_train_list.append(stats["mem_train"])
        mem_test_list.append(mem_test)

    print("\n=== Average over seeds ===")
    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)
    mean_ap = np.mean(ap_list)
    std_ap = np.std(ap_list)

    result_line = (
        f"{single_seed_prefix}"
        f"AUC: {mean_auc:.4f} +/- {std_auc:.4f} | "
        f"AP: {mean_ap:.4f} +/- {std_ap:.4f}"
    )
    print(result_line)

    mean_time_train = np.mean(time_train_list)
    mean_time_test = np.mean(time_test_list)
    mean_time_all = mean_time_train + mean_time_test
    mean_mem_train = np.mean(mem_train_list)
    mean_mem_test = np.mean(mem_test_list)

    std_time_train = np.std(time_train_list)
    std_time_test = np.std(time_test_list)
    std_time_all = np.std([train + test for train, test in zip(time_train_list, time_test_list)])
    std_mem_train = np.std(mem_train_list)
    std_mem_test = np.std(mem_test_list)

    print("\n=== Runtime over seeds ===")
    print(f"Time (Train): {mean_time_train:.4f} +/- {std_time_train:.4f}s")
    print(f"Mem (Train): {mean_mem_train / 1024 / 1024:.4f} +/- {std_mem_train / 1024 / 1024:.4f} MB")
    print(f"Time (Test): {mean_time_test:.4f} +/- {std_time_test:.4f}s")
    print(f"Mem (Test): {mean_mem_test / 1024 / 1024:.4f} +/- {std_mem_test / 1024 / 1024:.4f} MB")
    print(f"Time (all): {mean_time_all:.4f} +/- {std_time_all:.4f}s")

    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write(
            f"\n=== Result over seeds "
            f"[seeds={seed_text}, alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}, "
            f"lr={args.lr}, weight_decay={args.weight_decay}, bn={args.bn}, "
            f"num_epoch={args.num_epoch}, "
        )
        f.write(result_line + "\n")
        f.write(f"Time (Train): {mean_time_train:.4f} +/- {std_time_train:.4f}s\n")
        f.write(f"Mem (Train): {mean_mem_train / 1024 / 1024:.4f} +/- {std_mem_train / 1024 / 1024:.4f} MB\n")
        f.write(f"Time (Test): {mean_time_test:.4f} +/- {std_time_test:.4f}s\n")
        f.write(f"Mem (Test): {mean_mem_test / 1024 / 1024:.4f} +/- {std_mem_test / 1024 / 1024:.4f} MB\n")
        f.write(f"Time (all): {mean_time_all:.4f} +/- {std_time_all:.4f}s\n")
        f.write("=" * 40 + "\n")
