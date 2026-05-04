#!/usr/bin/env bash
# Run every auxiliary experiment family in sequence, suitable for unattended
# overnight execution.  All output lands under data/aux_<family>/ with one
# combined log at data/runAllExperiments_<timestamp>.log.
#
# Usage:
#   scripts/runAllExperiments.sh [profile]
#
# profile:
#   quick   tiny sanity-check sweep   (~10-15 minutes total)
#   medium  reduced-scale sweep       (~1-2 hours, default)
#   full    report-protocol sweep     (~6-9 hours)
#
# Environment overrides (any of these wins over the profile):
#   N_BATCHES, BATCH_SIZE, MAX_STEPS, SEEDS, EVAL_EPISODES, ROTOR_COUNTS,
#   AXES, KNOBS, FAILURE_ROTOR_COUNTS, OBS_REGIMES
#
# All families resume from previously-completed runs (the shared subprocess
# helper checks for summary.json + actor_*.pth before re-launching).

set -u  # error on unset vars; do NOT set -e so one family failing doesn't
        # abort the whole backlog.

PROFILE="${1:-medium}"

# --- Resolve repo root ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${REPO_ROOT}"

# --- Pick Python ---
PY="${FLEDGLING_PY:-python}"
# Prefer the project conda env if it exists.
if [ -x "$HOME/anaconda3/envs/fledgling-rl/bin/python" ]; then
    PY="$HOME/anaconda3/envs/fledgling-rl/bin/python"
fi

# --- Profile defaults ---
case "${PROFILE}" in
    quick)
        : "${N_BATCHES:=5}"
        : "${BATCH_SIZE:=256}"
        : "${MAX_STEPS:=200}"
        : "${SEEDS:=1}"
        : "${EVAL_EPISODES:=1}"
        : "${ROTOR_COUNTS:=4}"
        : "${AXES:=VERT}"
        : "${KNOBS:=lr,gamma}"
        : "${FAILURE_ROTOR_COUNTS:=4}"
        : "${OBS_REGIMES:=full,inertial_only}"
        ;;
    medium)
        : "${N_BATCHES:=30}"
        : "${BATCH_SIZE:=512}"
        : "${MAX_STEPS:=300}"
        : "${SEEDS:=2}"
        : "${EVAL_EPISODES:=3}"
        : "${ROTOR_COUNTS:=4,6,8}"
        : "${AXES:=VERT,PITCHLON,ROLLLAT,YAWHDG}"
        : "${KNOBS:=lr,gamma,lam,clip_eps,entropy_coeff,hidden_dim,batch_size,vel_penalty,overshoot_weight,rotor_balance,energy}"
        : "${FAILURE_ROTOR_COUNTS:=4,6,8}"
        : "${OBS_REGIMES:=full,inertial_only,position_only,noisy_full,delayed_full,inertial_only_stacked,delayed_full_stacked}"
        ;;
    full)
        : "${N_BATCHES:=200}"
        : "${BATCH_SIZE:=2048}"
        : "${MAX_STEPS:=500}"
        : "${SEEDS:=3}"
        : "${EVAL_EPISODES:=5}"
        : "${ROTOR_COUNTS:=4,6,8}"
        : "${AXES:=VERT,PITCHLON,ROLLLAT,YAWHDG}"
        : "${KNOBS:=lr,gamma,lam,clip_eps,entropy_coeff,hidden_dim,batch_size,vel_penalty,overshoot_weight,rotor_balance,energy}"
        : "${FAILURE_ROTOR_COUNTS:=4,6,8}"
        : "${OBS_REGIMES:=full,inertial_only,position_only,noisy_full,delayed_full,inertial_only_stacked,delayed_full_stacked}"
        ;;
    *)
        echo "Unknown profile: ${PROFILE}.  Use one of: quick | medium | full" >&2
        exit 1
        ;;
esac

# --- Logging ---
TS="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE="data/runAllExperiments_${TS}.log"
mkdir -p data
exec > >(tee -a "${LOG_FILE}") 2>&1

# --- KMP fix is a no-op on Linux but required on macOS ---
export KMP_DUPLICATE_LIB_OK=TRUE
export MPLBACKEND=Agg

echo "============================================================"
echo "Fledgling auxiliary-experiment backlog runner"
echo "Profile         : ${PROFILE}"
echo "Repo root       : ${REPO_ROOT}"
echo "Python          : ${PY}"
echo "Log             : ${LOG_FILE}"
echo "Started         : $(date)"
echo "------------------------------------------------------------"
echo "n_batches       = ${N_BATCHES}"
echo "batch_size      = ${BATCH_SIZE}"
echo "max_steps       = ${MAX_STEPS}"
echo "seeds           = ${SEEDS}"
echo "eval_episodes   = ${EVAL_EPISODES}"
echo "rotor_counts    = ${ROTOR_COUNTS}"
echo "axes            = ${AXES}"
echo "knobs           = ${KNOBS}"
echo "failure_rotors  = ${FAILURE_ROTOR_COUNTS}"
echo "obs_regimes     = ${OBS_REGIMES}"
echo "============================================================"

run_family () {
    local family="$1"; shift
    echo ""
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] >>> Running family: ${family}"
    echo "============================================================"
    local t0=$SECONDS
    if "$@"; then
        local dt=$(( SECONDS - t0 ))
        echo "[$(date +%H:%M:%S)] <<< ${family} OK in ${dt}s"
    else
        local rc=$?
        local dt=$(( SECONDS - t0 ))
        echo "[$(date +%H:%M:%S)] <<< ${family} FAILED (exit ${rc}) after ${dt}s"
    fi
}

# Common args shared across runners
COMMON_ARGS=(
    --n_batches    "${N_BATCHES}"
    --batch_size   "${BATCH_SIZE}"
    --max_steps    "${MAX_STEPS}"
    --seeds        "${SEEDS}"
    --eval_episodes "${EVAL_EPISODES}"
)

# --- Section 5.2: cross-config (rotor counts x axes) ---
run_family "5.2 cross-config" \
    "${PY}" scripts/runCrossConfigExperiments.py \
        --out_dir data/aux_crossconfig \
        --rotor_counts "${ROTOR_COUNTS}" \
        --axes "${AXES}" \
        "${COMMON_ARGS[@]}"

# --- Section 5.3a: 2x2 robustness ablation ---
run_family "5.3a robustness 2x2" \
    "${PY}" scripts/runRobustnessExperiments.py \
        --out_dir data/aux_robustness \
        --skip_failure \
        "${COMMON_ARGS[@]}"

# --- Section 5.3b: rotor-failure across rotor counts ---
run_family "5.3b rotor failure" \
    "${PY}" scripts/runRobustnessExperiments.py \
        --out_dir data/aux_robustness \
        --skip_2x2 \
        --failure_rotor_counts "${FAILURE_ROTOR_COUNTS}" \
        "${COMMON_ARGS[@]}"

# --- Section 5.4: restricted observability ---
run_family "5.4 observability" \
    "${PY}" scripts/runObservabilityExperiments.py \
        --out_dir data/aux_observability \
        --regimes "${OBS_REGIMES}" \
        "${COMMON_ARGS[@]}"

# --- Section 5.5: hyperparameter sensitivity ---
run_family "5.5 hparam sweep" \
    "${PY}" scripts/runHyperparameterSweep.py \
        --out_dir data/aux_hparams \
        --knobs "${KNOBS}" \
        "${COMMON_ARGS[@]}"

# --- Aggregate plots ---
run_family "aggregation + plots" \
    "${PY}" scripts/plotAuxiliaryResults.py \
        --robustness_json    data/aux_robustness/robustness_all.json \
        --observability_json data/aux_observability/observability_results.json \
        --out_dir            data/aux_plots

echo ""
echo "============================================================"
echo "All families processed.  Total wall time: ${SECONDS}s"
echo "Finished        : $(date)"
echo "Log             : ${LOG_FILE}"
echo "Summaries:"
ls data/aux_*/   2>/dev/null | grep -E "_summary\.md$|_results\.json$|_all\.json$" || true
echo "Plots:"
ls data/aux_plots/ 2>/dev/null || true
echo "============================================================"
