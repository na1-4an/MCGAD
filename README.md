


# MCGAD
> **MCGAD** (one-class **M**onophily-based **C**ontrastive learning for **G**raph **A**nomaly **D**etection)

---
## 1. Overview
This is an anonymized repository for paper **[One-class Monophily-Aware Contrastive Learning for Graph Anomaly Detection]** submitted to DASFAA 2026.


## 2. Repository Structure

```
repo_root/
  ├── modules/                
  │   ├── experiment.py      
  │   ├── model.py           
  │   ├── train.py       
  │   └── utils.py             
  ├── 2_hop_map/                
  ├── 2hop_aggre/    
  ├── dataset/             
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

| Dataset        | α   | β   | Learning Rate | Weight Decay | BatchNorm | Epochs  |
| -------------- | --- | --- | ------------- | ------------ | --------- | ------- |
| **Book**       | 0.2 | 0.3 | 0.0003        | 0            | ✅         | 100 |
| **Reddit**     | 0.4 | 0.1 | 0.0001        | 0            | ✅         | 100 |
| **Amazon-all** | 0.1 | 0.4 | 0.0003        | 0.0005       | –         | 100 |
| **Tolokers**   | 0.9 | 0.7 | 0.0001        | 0.0005       | ✅         | 100 |
| **T-Finance**  | 0.7 | 1.1 | 0.0005        | 0.0005       | ✅         | 500 |
| **Elliptic**   | 0.9 | 1.1 | 0.0005        | 0            | –         | 500 |
| **YelpChi**    | 0.1 | 0.1 | 0.0001        | 0            | –         | 100 |
| **Questions**  | 0.2 | 0.1 | 0.0005        | 0            | –         | 500 |
