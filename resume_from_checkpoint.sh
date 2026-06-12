#!/bin/bash
# Resume a sweep from its checkpoints by chaining dependent SLURM jobs.
#
# A GPU-shared job is walltime-capped (48h here), often not enough for a full
# grid sweep. This submits N jobs that all share the sweep's INVOCATION and run
# back-to-back (each `afterany` the previous), so each one:
#   * skips finished grid runs (their final/ dir already exists),
#   * resumes the in-progress run from its latest checkpoint-*,
#   * reuses the cached bios bins (.data_ready fingerprint) — no re-tokenize.
# When the sweep finishes, any leftover job starts, finds every run done, and
# exits within ~1 min.
#
# Pass --after <JID> to make the FIRST job wait on an already-running job, so
# you can queue the continuation now and it fires the moment that job ends.
#
# Usage:
#   ./resume_from_checkpoint.sh INVOCATION [N_JOBS] [--after RUNNING_JID] [SCRIPT]
# Examples:
#   ./resume_from_checkpoint.sh 20260610-211512 2 --after 41285633
#   ./resume_from_checkpoint.sh 20260610-211512        # 2 jobs, start now
set -euo pipefail

INVOCATION=${1:?usage: resume_from_checkpoint.sh INVOCATION [N_JOBS] [--after JID] [SCRIPT]}
shift

N_JOBS=2
AFTER=""
SCRIPT=ablation_llama_grid_robustness.py
while [ $# -gt 0 ]; do
    case "$1" in
        --after) AFTER=$2; shift 2 ;;
        *) if [[ "$1" =~ ^[0-9]+$ ]]; then N_JOBS=$1; else SCRIPT=$1; fi; shift ;;
    esac
done

echo "Resume INVOCATION=$INVOCATION  script=$SCRIPT  chain=$N_JOBS${AFTER:+  after job $AFTER}"

dep="$AFTER"
for i in $(seq 1 "$N_JOBS"); do
    a=(--parsable --export=ALL,INVOCATION="$INVOCATION")
    [ -n "$dep" ] && a+=(--dependency=afterany:"$dep")
    jid=$(sbatch "${a[@]}" setup_job_psc.sh "$SCRIPT")
    echo "  [$i/$N_JOBS] submitted $jid${dep:+  (afterany $dep)}"
    dep=$jid
done

echo "Watch: squeue -u \$USER"
