# Physics-Guided Recurrent State-Space Neural Networks for Multi-Step Prediction

IFAC World Congress 2026 — Submission #1679

This repository contains the code to reproduce all results in the paper.
Each experiment section below lists the exact commands used to generate the paper's results,
with logdir names matching the original run structure.

## Installation

```bash
pip install -r requirements.txt
```

## Model Name Mapping

| Paper name | `--model` | `--mpnt_wt` |
|------------|-----------|-------------|
| RSSNN | `AE-RNN-U` | `-1` |
| PG-RSSNN | `AE-RNN-U` | `10` |
| RSSNN-sgm | `AE-RNN-U-SGM` | `-1` |
| PG-RSSNN-sgm | `AE-RNN-U-SGM` | `10` |
| RSSNN-fb | `AE-RNN-XU` | `-1` |
| RSSNN (no input) | `AE-RNN` | `-1` |
| PG-RSSNN (no input) | `AE-RNN` | `10` |
| FC | `MLP-U` | `-1` |
| PG-FC | `MLP-U` | `10` |
| FC-sgm | `MLP-U-SGM` | `-1` |
| PG-FC-sgm | `MLP-U-SGM` | `10` |
| Liu et al. | `LIU-U` | `-1` or `10` |
| Physics-only | `AE-RNN-U` | `1000` |

**`--mpnt_wt` → filename suffix:** `-1` → `mpw-10`, `10` → `mpw100`, `1000` → `mpw10000`

---

## Experiment 1 — Toy LGSSM (`0813_full_100`)

50 rounds, full data (`k_max_train=2000`, `seq_len=64`).

`--A_prt_idx`, `--B_prt_idx`, `--C_prt_idx`: physics knowledge level — `0`=none, `1`=approximate, `2`=exact.

```bash
# --- AE-RNN-U ---
python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=-1  --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1

python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=1 --B_prt_idx=0 --C_prt_idx=0

python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1

python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=2 --B_prt_idx=0 --C_prt_idx=0

python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=2 --B_prt_idx=2 --C_prt_idx=2

# --- AE-RNN ---
python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=-1  --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1

python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1

python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=2 --B_prt_idx=0 --C_prt_idx=0

python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=2 --B_prt_idx=2 --C_prt_idx=2

# --- AE-RNN-XU ---
python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-XU --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=-1  --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1

python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-XU --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1

python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-XU --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=2 --B_prt_idx=0 --C_prt_idx=0

python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-XU --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=2 --B_prt_idx=2 --C_prt_idx=2

# --- MLP-U ---
python main_single.py --dataset=toy_lgssm_5_pre --model=MLP-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=-1  --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1

python main_single.py --dataset=toy_lgssm_5_pre --model=MLP-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1

python main_single.py --dataset=toy_lgssm_5_pre --model=MLP-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=2 --B_prt_idx=0 --C_prt_idx=0

python main_single.py --dataset=toy_lgssm_5_pre --model=MLP-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=2 --B_prt_idx=2 --C_prt_idx=2

# --- LIU-U ---
python main_single.py --dataset=toy_lgssm_5_pre --model=LIU-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=-1  --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1

python main_single.py --dataset=toy_lgssm_5_pre --model=LIU-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1

python main_single.py --dataset=toy_lgssm_5_pre --model=LIU-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=2 --B_prt_idx=0 --C_prt_idx=0

python main_single.py --dataset=toy_lgssm_5_pre --model=LIU-U --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=2 --B_prt_idx=2 --C_prt_idx=2

# --- AE-RNN-U-SGM ---
python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-U-SGM --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=-1  --A_prt_idx=2 --B_prt_idx=0 --C_prt_idx=0

python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-U-SGM --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1

python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-U-SGM --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=2 --B_prt_idx=0 --C_prt_idx=0

# --- MLP-U-SGM ---
python main_single.py --dataset=toy_lgssm_5_pre --model=MLP-U-SGM --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=-1  --A_prt_idx=2 --B_prt_idx=0 --C_prt_idx=0

python main_single.py --dataset=toy_lgssm_5_pre --model=MLP-U-SGM --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=50 --mpnt_wt=10 --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1
```

---

## Experiment 2 — Cascaded Tank

`seq_len_train` must always equal `k_max_train` (training data = one sequence).

### Full data — all models (20 rounds)

```bash
for MODEL in AE-RNN-U AE-RNN AE-RNN-XU MLP-U LIU-U; do
  python main_single.py --dataset=cascaded_tank --model=$MODEL --do_train --do_test \
    --logdir=0813_full_100 --train_rounds=20 --mpnt_wt=-1
  python main_single.py --dataset=cascaded_tank --model=$MODEL --do_train --do_test \
    --logdir=0813_full_100 --train_rounds=20 --mpnt_wt=10
done

# SGM variants (10 rounds)
for MODEL in AE-RNN-U-SGM MLP-U-SGM; do
  python main_single.py --dataset=cascaded_tank --model=$MODEL --do_train --do_test \
    --logdir=0813_full_100 --train_rounds=10 --mpnt_wt=-1
  python main_single.py --dataset=cascaded_tank --model=$MODEL --do_train --do_test \
    --logdir=0813_full_100 --train_rounds=10 --mpnt_wt=10
done
```

### Data efficiency — AE-RNN-U and AE-RNN only (10 rounds)

```bash
# 50% data  (k_max=seq_len=512)
for MODEL in AE-RNN-U AE-RNN; do
  python main_single.py --dataset=cascaded_tank --model=$MODEL --do_train --do_test \
    --logdir=0813_full_50 --train_rounds=10 --k_max_train=512 --seq_len_train=512 --mpnt_wt=-1
  python main_single.py --dataset=cascaded_tank --model=$MODEL --do_train --do_test \
    --logdir=0813_full_50 --train_rounds=10 --k_max_train=512 --seq_len_train=512 --mpnt_wt=10
done

# 20% data  (k_max=seq_len=256)
for MODEL in AE-RNN-U AE-RNN; do
  python main_single.py --dataset=cascaded_tank --model=$MODEL --do_train --do_test \
    --logdir=0813_full_20 --train_rounds=10 --k_max_train=256 --seq_len_train=256 --mpnt_wt=-1
  python main_single.py --dataset=cascaded_tank --model=$MODEL --do_train --do_test \
    --logdir=0813_full_20 --train_rounds=10 --k_max_train=256 --seq_len_train=256 --mpnt_wt=10
done

# 10% data  (k_max=seq_len=128, ~17% of training set)
for MODEL in AE-RNN-U AE-RNN; do
  python main_single.py --dataset=cascaded_tank --model=$MODEL --do_train --do_test \
    --logdir=0813_full_10 --train_rounds=10 --k_max_train=128 --seq_len_train=128 --mpnt_wt=-1
  python main_single.py --dataset=cascaded_tank --model=$MODEL --do_train --do_test \
    --logdir=0813_full_10 --train_rounds=10 --k_max_train=128 --seq_len_train=128 --mpnt_wt=10
done
```

---

## Experiment 3 — Industrial Robot Simulation

`seq_len=606` fixed. `--k_max_train` is the fraction of simulation data (0–1).

### Full data — all models (10 rounds)

```bash
for MODEL in AE-RNN-U AE-RNN AE-RNN-XU AE-RNN-U-SGM MLP-U MLP-U-SGM LIU-U; do
  python main_single.py --dataset=industrobo --model=$MODEL --do_train --do_test \
    --logdir=0813_full_100 --train_rounds=10 --if_simulation=1 --k_max_train=1.0 --mpnt_wt=-1
  python main_single.py --dataset=industrobo --model=$MODEL --do_train --do_test \
    --logdir=0813_full_100 --train_rounds=10 --if_simulation=1 --k_max_train=1.0 --mpnt_wt=10
done

# Physics-only baseline (AE-RNN only, 10 rounds)
python main_single.py --dataset=industrobo --model=AE-RNN --do_train --do_test \
  --logdir=0813_full_100 --train_rounds=10 --if_simulation=1 --k_max_train=1.0 --mpnt_wt=1000
```


---

## Experiment 4 — Industrial Robot Real Data

Requires `data/IndustRobo/forward_identification_without_raw_data_shifted.mat`.  
`seq_len=606` (605 used internally for the shifted real data).

```bash
# AE-RNN-U-SGM — main proposed model (10 rounds)
python main_single.py --dataset=industrobo --model=AE-RNN-U-SGM --do_train --do_test \
  --logdir=1010_robo_real_multi --train_rounds=10 --if_simulation=0 --k_max_train=1.0 --mpnt_wt=-1
python main_single.py --dataset=industrobo --model=AE-RNN-U-SGM --do_train --do_test \
  --logdir=1010_robo_real_multi --train_rounds=10 --if_simulation=0 --k_max_train=1.0 --mpnt_wt=10

# MLP-U (10 rounds)
python main_single.py --dataset=industrobo --model=MLP-U --do_train --do_test \
  --logdir=1010_robo_real_multi --train_rounds=10 --if_simulation=0 --k_max_train=1.0 --mpnt_wt=-1

# MLP-U-SGM (10 rounds)
python main_single.py --dataset=industrobo --model=MLP-U-SGM --do_train --do_test \
  --logdir=1010_robo_real_multi --train_rounds=10 --if_simulation=0 --k_max_train=1.0 --mpnt_wt=-1
python main_single.py --dataset=industrobo --model=MLP-U-SGM --do_train --do_test \
  --logdir=1010_robo_real_multi --train_rounds=10 --if_simulation=0 --k_max_train=1.0 --mpnt_wt=10

# LIU-U (10 rounds)
python main_single.py --dataset=industrobo --model=LIU-U --do_train --do_test \
  --logdir=1010_robo_real_multi --train_rounds=10 --if_simulation=0 --k_max_train=1.0 --mpnt_wt=-1
python main_single.py --dataset=industrobo --model=LIU-U --do_train --do_test \
  --logdir=1010_robo_real_multi --train_rounds=10 --if_simulation=0 --k_max_train=1.0 --mpnt_wt=10

# AE-RNN-XU (10 rounds)
python main_single.py --dataset=industrobo --model=AE-RNN-XU --do_train --do_test \
  --logdir=1010_robo_real_multi --train_rounds=10 --if_simulation=0 --k_max_train=1.0 --mpnt_wt=-1

# AE-RNN-U physics-only baseline (1 round)
python main_single.py --dataset=industrobo --model=AE-RNN-U --do_train --do_test \
  --logdir=1010_robo_real_multi --train_rounds=1 --if_simulation=0 --k_max_train=1.0 --mpnt_wt=1000
```
