


# MCGAD
> **MCGAD** (one-class **M**onophily-based **C**ontrastive learning for **G**raph **A**nomaly **D**etection)

---
## 1. Overview
This is a repository for paper **[One-class Monophily-Aware Contrastive Learning for Graph Anomaly Detection]**.


## 2. Repository Structure

```
repo_root/
  ├── modules/                
  │   ├── experiment.py      
  │   ├── model.py           
  │   ├── train.py       
  │   └── utils.py             
  ├── 2_hop_map/                
  ├── 2hop_standard_aggr/    
  ├── dataset/             
  ├── xl2x/             
  ├── requirements.txt       
  ├── run.py   
  └── run.sh       
```

---

## 3. Dataset
You can obtain dataset/, 2_hop_map/, 2hop_aggre/ from https://drive.google.com/drive/folders/1lKf3KB2fWLCVIgwZE0NEjQfyrnrpHrSl?usp=sharing


## 4. Quick Start

```bash
bash run.sh
```

## 5. Dataset-Specific Settings

The following table summarizes the hyperparameter settings used for each dataset:

| Dataset        | α   | β   | γ   | Learning Rate | BatchNorm |
| -------------- | --- | --- | --- | ------------- | --------- |
| **Book**       | 0.3 | 1.0 | 0.5 | 0.0001        | ✅         |
| **Reddit**     | 0.1 | 0.9 | 0.2 | 0.0005        | ✅         |
| **Amazon-all** | 0.1 | 0.8 | 0.5 | 0.001         | –         |
| **Tolokers**   | 0.2 | 0.1 | 0.5 | 0.0001        | –         |
| **T-Finance**  | 0.1 | 0.8 | 0.2 | 0.0001        | –         |
| **Elliptic**   | 0.1 | 1.0 | 0.5 | 0.001         | –         |
| **YelpChi**    | 0.1 | 0.7 | 0.5 | 0.0001        | ✅         |
| **Questions**  | 0.1 | 0.4 | 0.4 | 0.001         | ✅         |
