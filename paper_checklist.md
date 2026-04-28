# Paper Code & Configuration Checklist
**Paper:** Physics-Guided Recurrent State-Space Neural Networks for Multi-Step Prediction  
**Repo:** `yifang0-0/Physics-Guided-Recurrent-State-Space-Neural-Networks`  
**Status legend:** ✅ present & verified | ⚠️ present but needs attention | ❌ missing

---

## Current Progress

| Task | Status |
|------|--------|
| All 7 models in pg-rnn | ✅ |
| All 3 datasets wired up | ✅ |
| Default hyperparams match paper logs | ✅ |
| CLI arg aliases (`--n_epoch`, `--init_nr`) | ✅ fixed |
| Toy LGSSM reproducibility (rounds 1–4) | ✅ bit-for-bit identical |
| Simulation robot data files (10 × .mat) | ✅ |
| Real robot data file (4 MB) | ✅ included |
| `requirements.txt` | ✅ generated |
| IndustRobo sim verification (10 epochs) | ⚠️ **running** — PID 699845, log: `verify_robosim_out.log` |
| Cascaded Tank verification | ❌ not started |
| Investigate real robot dataset generation | ⚠️ see section 9 |

---

## 1. Models

| Model | File | Status | Notes |
|-------|------|--------|-------|
| AE-RNN | `models/model_ae_rnn.py` | ✅ | Base model, no input `u` |
| AE-RNN-U | `models/model_ae_rnn_u.py` | ✅ | **Main paper model** |
| AE-RNN-U-SGM | `models/model_ae_rnn_u_sgm.py` | ✅ | Structured gradient matching variant |
| AE-RNN-XU | `models/model_ae_rnn_xu.py` | ✅ | Extended input variant |
| MLP-U | `models/model_mlp_u.py` | ✅ | MLP baseline |
| MLP-U-SGM | `models/model_mlp_u_sgm.py` | ✅ | MLP + structured gradient matching |
| LIU-U | `models/model_liu_u.py` | ✅ | Liu et al. comparison baseline |

### mpnt_wt convention

| Value | Behaviour | CLI example | Filename suffix |
|-------|-----------|-------------|-----------------|
| `<= 0` (e.g. `-1`) | Black-box pure NN | `--mpnt_wt=-1` | `mpw-10` |
| `>= 10` (e.g. `10`) | Physics-guided (PG-) | `--mpnt_wt=10` | `mpw100` |
| `> 100` (e.g. `1000`) | Physics-only | `--mpnt_wt=1000` | `mpw10000` |

### Physical model

| Component | File | Status |
|-----------|------|--------|
| Physical model interface | `models/physical_augment/model_phy.py` | ✅ |
| KUKA300 DH parameters | `models/physical_augment/kuka300.py` | ✅ |
| `roboticstoolbox-python==1.1.1` | system | ✅ installed |

---

## 2. Datasets

### 2a. Toy LGSSM (`toy_lgssm_5_pre`)

| Item | Status |
|------|--------|
| Data loader | `data/toy_lgssm_5_pre.py` ✅ |
| Training sets | `data/Toy_LGSSM/toy_lgssm_pre_trainingset_shifted_{0..49}.npz` — 51 files ✅ |
| Test set | `data/Toy_LGSSM/toy_lgssm_pre_testset_shifted.npz` ✅ |
| Reproducibility | Rounds 1–4 bit-for-bit identical vs `0813_full_100` ✅ |

Config: `n_epochs=200, init_lr=1e-3, seq_len=64, h_dim=10, z_dim=2, n_layers=1`

### 2b. Cascaded Tanks (`cascaded_tank`)

| Item | Status |
|------|--------|
| Data loader | `data/cascTank.py` ✅ |
| Data source | `nonlinear-benchmarks==1.0.0` pip package — no files needed ✅ |
| Reproducibility | ❌ not yet verified |

Config: `n_epochs=1000, init_lr=1e-3, k_max_train=768, seq_len_train=768, h_dim=60, z_dim=2, n_layers=1`

### 2c. Industrial Robot — Simulation (`industrobo`, `if_simulation=1`)

| Item | Status |
|------|--------|
| Data loader | `data/IndustRobo.py` ✅ |
| Simulation .mat files | `data/IndustRobo/1010m_vary_sim_position_noiseTorque{0..9}.mat` — 10 files ✅ |
| KUKA config | `data/IndustRobo/kuka_config_model.py` ✅ |
| Reproducibility | ⚠️ running — `verify_robosim.py` (PID 699845) |

Config: `n_epochs=200, init_lr=1e-2, k_max_train=1.0, y_dim=6, u_dim=6, h_dim=64, z_dim=12, n_layers=3, if_G=1, if_clip=1, if_level2=1, if_level0=1, roboname=KUKA300, seq_len=606`

### 2d. Industrial Robot — Real Data (`industrobo`, `if_simulation=0`)

| Item | Status |
|------|--------|
| Real data file | `data/IndustRobo/forward_identification_without_raw_data_shifted.mat` — 4 MB ✅ |
| Reference results | `DeepSSM_SysID/log/1010_robo_real_multi/industrobo/` (original repo) |
| Reproducibility | ❌ not yet verified |

Config: `n_epochs=200, init_lr=1e-2, if_simulation=0, k_max_train=1.0` (same model config as sim)

---

## 3. Training Infrastructure

| Component | File | Status |
|-----------|------|--------|
| Main entry point | `main_single.py` | ✅ |
| Training loop | `training.py` | ✅ |
| Testing / evaluation | `testing.py` | ✅ |
| Dataset loader | `data/loader.py` | ✅ |
| Model registry | `models/dynamic_model.py` | ✅ |
| Hyperparameters | `options/train_options.py` | ✅ |
| Metrics (RMSE, NRMSE, VAF, corr) | `utils/dataevaluater.py` | ✅ |

---

## 4. Dependencies (`requirements.txt`)

```
torch==2.0.1
numpy==1.25.2
scipy==1.11.1
pandas==2.0.3
matplotlib==3.7.2
roboticstoolbox-python==1.1.1
nonlinear-benchmarks==1.0.0
scikit-learn==1.2.2
dtaidistance==2.3.12
tabulate==0.9.0
```

---

## 5. Default Hyperparameters (`options/train_options.py`) — verified vs `0813_full_100`

### Training options

| Dataset | n_epochs | init_lr | min_lr | lr_sched_epochs | lr_sched_factor |
|---------|----------|---------|--------|-----------------|-----------------|
| `cascaded_tank` | 1000 | 1e-3 | 1e-9 | 10 | 10 |
| `toy_lgssm_5_pre` | 200 | 1e-3 | 1e-7 | 30 | 5 |
| `industrobo` | 200 | **1e-2** | 1e-7 | 20 | 10 |

### Model options (h_dim fixed 2025-04-28)

| Dataset | h_dim | z_dim | n_layers | Paper says |
|---------|-------|-------|----------|------------|
| `cascaded_tank` | **16** (was 60 ❌) | 2 | 1 | n_h=16 ✅ |
| `toy_lgssm_5_pre` | **16** (was 10 ❌) | 2 | 1 | n_h=16 ✅ |
| `industrobo` | 64 | 12 | 3 | n_h=64 ✅ |

### Sequence lengths and data fractions

**Only cascaded tank** couples `seq_len_train = k_max_train` (entire training data = one long sequence).
For partial-data runs both must be set together. IndustRobo and LGSSM have fixed seq_len.

| Dataset | seq_len_train | k_max_train | seq_len fixed? |
|---------|--------------|-------------|----------------|
| `cascaded_tank` (10%) | **128** | **128** | No — must match |
| `cascaded_tank` (20%) | **256** | **256** | No — must match |
| `cascaded_tank` (50%) | **512** | **512** | No — must match |
| `cascaded_tank` (100%) | 768 (default) | 768 (default) | — |
| `industrobo` (all fractions) | 606 (fixed) | 0.1 / 0.2 / 0.5 / 1.0 | Yes |
| `toy_lgssm_5_pre` (all fractions) | 64 (fixed) | 200 / 400 / 1000 / 2000 | Yes |

> Note: `0813_full_10/toy_lgssm_5_pre` has `k_max_train=2000` (same as full_100) — that run was not actually reduced, likely a bug in the original experiment.

---

## 6. Verification Status

| Experiment | Reference | Status |
|------------|-----------|--------|
| LGSSM AE-RNN-U black-box (mpw-10) | `0813_full_100/toy_lgssm_5_pre/` | ✅ rounds 1–4 exact match |
| LGSSM AE-RNN-U physics-guided (mpw100) | `0813_full_100/toy_lgssm_5_pre/` | ✅ rounds 1–4 exact match |
| IndustRobo sim AE-RNN-U (mpw-10, mpw100) | `0813_full_100/industrobo/` | ⚠️ running (10 epochs × 10 rounds via verify_robosim.py using full_50 as proxy) |
| Cascaded Tank | `0813_full_100/cascaded_tank/` | ❌ not started |

---

## 7. Open Items

- [ ] **Wait for** `verify_robosim_out.log` to finish and check PASS/FAIL
- [ ] Run Cascaded Tank verification
- [ ] Add `README.md` with example training commands
- [ ] Confirm `if_bias` (default=0, absent from paper runs) documented as post-paper feature
- [x] `requirements.txt` generated
- [x] Real robot data included (4 MB)
- [x] CLI arg aliases fixed (`--n_epoch`, `--init_nr`)

---

## 8. Example Training Commands (from original paper runs)

```bash
# Toy LGSSM — black-box (100% data)
python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-U \
  --do_train --do_test --logdir=0813_full_100 --mpnt_wt=-1 --train_rounds=5

# Toy LGSSM — physics-guided
python main_single.py --dataset=toy_lgssm_5_pre --model=AE-RNN-U \
  --do_train --do_test --logdir=0813_full_100 --mpnt_wt=10 --A_prt_idx=1 --B_prt_idx=1 --C_prt_idx=1 --train_rounds=5

# IndustRobo sim — black-box (50% training data, seq_len stays 606)
python main_single.py --dataset=industrobo --model=AE-RNN-U \
  --do_train --do_test --logdir=0813_full_50 --mpnt_wt=-1 --if_simulation=1 --k_max_train=0.5 --train_rounds=10

# Cascaded Tank — 100% data (seq_len_train = k_max_train = 768)
python main_single.py --dataset=cascaded_tank --model=AE-RNN-U \
  --do_train --do_test --logdir=0813_full_100 --mpnt_wt=-1 --train_rounds=10

# Cascaded Tank — 10% data (seq_len_train MUST equal k_max_train)
python main_single.py --dataset=cascaded_tank --model=AE-RNN-U \
  --do_train --do_test --logdir=0813_full_10 --mpnt_wt=-1 --train_rounds=10 \
  --k_max_train=128 --seq_len_train=128

# Cascaded Tank — 20% data
python main_single.py --dataset=cascaded_tank --model=AE-RNN-U \
  --do_train --do_test --logdir=0813_full_20 --mpnt_wt=-1 --train_rounds=10 \
  --k_max_train=256 --seq_len_train=256

# Cascaded Tank — 50% data
python main_single.py --dataset=cascaded_tank --model=AE-RNN-U \
  --do_train --do_test --logdir=0813_full_50 --mpnt_wt=-1 --train_rounds=10 \
  --k_max_train=512 --seq_len_train=512
```

---

## 9. Real Robot Dataset — Source and Preprocessing

**Original benchmark:** "Dataset and Baseline for an Industrial Robot Identification Benchmark"  
Weigand, Götz, Ulmen, Ruskowski — TU Kaiserslautern (2024)  
https://kluedo.ub.rptu.de/frontdoor/index/index/docId/7284

**Original file:** `forward_identification_without_raw_data.mat`  
- `u_train`: (6, 39988) — motor torques in **Nm**  
- `y_train`: (6, 39988) — joint positions in **deg**  
- `u_test`:  (6, 3636), `y_test`: (6, 3636)  
- No validation split in original

**Changes made (notebook: `Interactive-1.ipynb`):**

1. **Padding**: appended 8 samples to training data to make it divisible by 606 → 39996 = 66 × 606

2. **Train/val split**: split 66 sequences into 60 train + 6 val (90/10%)
   - train: (6, 36360), val: (6, 3636)

3. **Time alignment shift** (`shift_position_length(..., length=606)`):  
   For each 606-step sequence: removes first position step and last torque step, yielding 605-step sequences where `u[t]` predicts `y[t+1]`.  
   This enforces causal one-step-ahead alignment for forward dynamics.
   - After shift: train (6, 36300), val (6, 3630), test (6, 3630)

4. **Unit conversion (in data loader, not in file)**: positions converted deg → rad via `y / 180 * π` in `IndustRobo.py`. Torques stay in Nm.

5. **`seq_len -= 1` in `IndustRobo.py`** (line 13–15): accounts for the shift — passes `seq_len=605` to `IODataset` so batching is correct.

**Result:** `forward_identification_without_raw_data_shifted.mat` — 4 MB, included in repo.
