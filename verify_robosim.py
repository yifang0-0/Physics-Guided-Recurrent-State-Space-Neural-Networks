#!/usr/bin/env python3
"""
Verify pg-rnn produces identical per-epoch losses to the cached original runs.

Reproduces the 0813_full_50 command (k_max_train=0.5) for 10 epochs only,
comparing the first 2 loss entries (ep5, ep10) from each No0-No9 training
record against the cached 0813_full_50 records.

Equivalent original command that produced the reference:
  python main_single.py --dataset=industrobo --model=AE-RNN-U \
    --do_train --do_test --logdir=0813_full_50 --n_epoch=200 \
    --mpnt_wt=-1 --if_simulation=1 --k_max_train=0.5 --train_rounds=10

Usage (background):
  cd /home/ruiyuanli/dcscgpuserver1/pg-rnn
  nohup python verify_robosim.py > verify_robosim_out.log 2>&1 &
"""

import subprocess
import json
import os
import sys

PGRNN_DIR    = "/home/ruiyuanli/dcscgpuserver1/pg-rnn"
ORIG_BASE    = "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log"
REF_LOGDIR   = "0813_full_50"       # cached reference run to compare against
ORIG_DIR     = os.path.join(ORIG_BASE, REF_LOGDIR, "industrobo", "AE-RNN-U_None")
NEW_LOGDIR   = "verify_robosim_50"  # separate from verify_pgvbb to avoid collision
TRAIN_ROUNDS = 10
N_EPOCHS     = 10    # produces 2 loss entries (ep5, ep10)
K_MAX_TRAIN  = 0.5   # must match REF_LOGDIR's k_max_train


def run_training(mpnt_wt):
    cmd = [
        "python", "main_single.py",
        "--dataset=industrobo",
        "--model=AE-RNN-U",
        "--do_train",
        f"--logdir={NEW_LOGDIR}",
        f"--n_epoch={N_EPOCHS}",
        f"--mpnt_wt={mpnt_wt}",
        "--if_simulation=1",
        f"--k_max_train={K_MAX_TRAIN}",
        f"--train_rounds={TRAIN_ROUNDS}",
    ]
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}", flush=True)
    result = subprocess.run(cmd, cwd=PGRNN_DIR)
    return result.returncode


def compare_records(mpnt_wt_float):
    mpw_val = int(mpnt_wt_float * 10)
    mpw_str = f"mpw{mpw_val}"

    print(f"\n{'='*60}")
    print(f"Comparing {mpw_str} (mpnt_wt={mpnt_wt_float})  ref={REF_LOGDIR}")
    print(f"{'='*60}")
    print(f"  {'Round':<8} {'Orig ep5':>14} {'New ep5':>14}  {'Orig ep10':>14} {'New ep10':>14}  Status")
    print(f"  {'-'*8} {'-'*14} {'-'*14}  {'-'*14} {'-'*14}  {'-'*15}")

    all_match = True
    for i in range(TRAIN_ROUNDS):
        fname     = f"industrobo_h64_z12_n3_{mpw_str}_No{i}_trainingrecord.json"
        new_path  = os.path.join(PGRNN_DIR, "log", NEW_LOGDIR, "industrobo", "AE-RNN-U_None", fname)
        orig_path = os.path.join(ORIG_DIR, fname)

        if not os.path.exists(orig_path):
            print(f"  No{i:<6}  MISSING reference: {orig_path}")
            all_match = False
            continue
        if not os.path.exists(new_path):
            print(f"  No{i:<6}  MISSING new output: {new_path}")
            all_match = False
            continue

        with open(new_path)  as f: new_rec  = json.load(f)
        with open(orig_path) as f: orig_rec = json.load(f)

        new_losses  = new_rec['all_losses'][:2]
        orig_losses = orig_rec['all_losses'][:2]

        diffs    = [abs(a - b) for a, b in zip(new_losses, orig_losses)]
        max_diff = max(diffs)
        match    = max_diff == 0.0

        status = "EXACT MATCH" if match else f"DIFF max={max_diff:.2e}"
        print(f"  No{i:<6}  {orig_losses[0]:>14.6f} {new_losses[0]:>14.6f}  "
              f"{orig_losses[1]:>14.6f} {new_losses[1]:>14.6f}  {status}")

        if not match:
            all_match = False

    return all_match


if __name__ == "__main__":
    print("=== Robosim Verification Script ===")
    print(f"Reference: {ORIG_DIR}")
    print(f"k_max_train={K_MAX_TRAIN}  n_epochs={N_EPOCHS}  train_rounds={TRAIN_ROUNDS}", flush=True)

    rc1 = run_training(-1)   # black-box   mpnt_wt=-1  -> mpw-10
    rc2 = run_training(10)   # phys-guided mpnt_wt=10  -> mpw100

    print("\n\n=== Training done, comparing results ===")

    match1 = compare_records(-1.0)
    match2 = compare_records(10.0)

    print(f"\n{'='*60}")
    print("FINAL RESULT:")
    print(f"  mpw-10  (black-box):      {'PASS' if match1 else 'FAIL'}")
    print(f"  mpw100  (physics-guided): {'PASS' if match2 else 'FAIL'}")
    print(f"{'='*60}")

    sys.exit(0 if (match1 and match2) else 1)
