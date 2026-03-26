#!/bin/bash
# ============================================================================
# Parameter Golf — Experiment Progress Monitor
# Run from anywhere: bash /path/to/experiments/check_progress.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/experiments/results"
PROGRESS_FILE="${RESULTS_DIR}/progress.json"

# ── Box drawing header ─────────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "       Parameter Golf -- Experiment Monitor              "
echo "========================================================"
echo ""

# ── Check if experiments are running ────────────────────────────────────────

if [ ! -f "${PROGRESS_FILE}" ]; then
    echo "STATUS: No experiments running."
    echo "Start with: bash experiments/run_all.sh"
    echo ""

    # Still show any completed results if they exist
    completed_count=$(find "${RESULTS_DIR}" -name '*.json' ! -name 'progress.json' 2>/dev/null | wc -l | tr -d ' ')
    if [ "${completed_count}" -gt 0 ]; then
        echo "(${completed_count} completed result files found — showing below)"
        echo ""
    else
        exit 0
    fi
else
    # Parse progress.json
    current=$(sed -n 's/.*"current": *"\([^"]*\)".*/\1/p' "${PROGRESS_FILE}")
    idx=$(sed -n 's/.*"index": *\([0-9]*\).*/\1/p' "${PROGRESS_FILE}")
    total=$(sed -n 's/.*"total": *\([0-9]*\).*/\1/p' "${PROGRESS_FILE}")
    start_time=$(sed -n 's/.*"start_time": *\([0-9]*\).*/\1/p' "${PROGRESS_FILE}")
    run_start_time=$(sed -n 's/.*"run_start_time": *\([0-9]*\).*/\1/p' "${PROGRESS_FILE}")

    now=$(date +%s)
    elapsed=$(( now - start_time ))
    total_elapsed=""
    if [ -n "${run_start_time}" ]; then
        total_elapsed=$(( now - run_start_time ))
    fi

    # Get latest loss from log tail
    log_file="${RESULTS_DIR}/${current}.log"
    latest_loss=""
    latest_step=""
    if [ -f "${log_file}" ]; then
        last_train_line=$(grep -E 'step:[0-9]+.*train_loss:' "${log_file}" | tail -1 || true)
        if [ -n "${last_train_line}" ]; then
            latest_loss=$(echo "${last_train_line}" | sed -E 's/.*train_loss:([0-9.]+).*/\1/')
            latest_step=$(echo "${last_train_line}" | sed -E 's/.*step:([0-9]+)\/.*/\1/')
        fi
    fi

    # Status line
    if [ -n "${total}" ] && [ "${total}" -gt 0 ]; then
        echo "STATUS: Running ${current} (${idx}/${total})"
    else
        echo "STATUS: Running ${current}"
    fi

    status_detail="This experiment: ${elapsed}s"
    if [ -n "${total_elapsed}" ]; then
        total_min=$(( total_elapsed / 60 ))
        status_detail="Total run: ${total_min}m ${total_elapsed}s | This experiment: ${elapsed}s"
    fi
    if [ -n "${latest_loss}" ]; then
        status_detail="${status_detail} | Step: ${latest_step} | Loss: ${latest_loss}"
    fi
    echo "${status_detail}"

    # ETA estimates (~720s per training experiment)
    avg_per_exp=720
    if [ -n "${total}" ] && [ "${total}" -gt 0 ] && [ -n "${idx}" ]; then
        remaining_exp=$(( total - idx ))
        # Time left on current experiment (estimate)
        if [ ${elapsed} -lt ${avg_per_exp} ]; then
            eta_current=$(( avg_per_exp - elapsed ))
        else
            eta_current=0
        fi
        eta_remaining=$(( eta_current + remaining_exp * avg_per_exp ))
        eta_min=$(( eta_remaining / 60 ))
        echo "ETA this experiment: ~$(( eta_current / 60 )) min | ETA all remaining: ~${eta_min} min"
    fi
    echo ""
fi

# ── Completed experiments ──────────────────────────────────────────────────

echo "COMPLETED EXPERIMENTS"
echo "---------------------"

# Display order
display_order=(
    v00_baseline
    eval_stride32
    eval_recurrence
    v01_xsa_all
    v02_rebalance_8kv
    v03_no_resid_mix_dec
    v04_trigram_conv
    v05_no_value_embed
    v08_engram_lite
    v09_dca_scalar
    v10_dca_full
    v06_12L_int5mlp
    v07_ushaped_quant
)

# Read baseline sliding BPB
baseline_sliding=""
if [ -f "${RESULTS_DIR}/v00_baseline.json" ]; then
    baseline_sliding=$(sed -n 's/.*"sliding_bpb": *\([0-9.]*\).*/\1/p' "${RESULTS_DIR}/v00_baseline.json")
fi

printf "%-24s| %-12s| %-11s| %s\n" "Experiment" "sliding_bpb" "D baseline" "artifact_mb"

has_results=0
for name in "${display_order[@]}"; do
    json_file="${RESULTS_DIR}/${name}.json"
    [ -f "${json_file}" ] || continue
    has_results=1

    sliding=$(sed -n 's/.*"sliding_bpb": *\([0-9.]*\).*/\1/p' "${json_file}")
    artifact_bytes=$(sed -n 's/.*"artifact_bytes": *\([0-9]*\).*/\1/p' "${json_file}")
    notes=$(sed -n 's/.*"notes": *"\([^"]*\)".*/\1/p' "${json_file}")

    slide_str="${sliding:---}"

    delta_str="--"
    if [ -n "${sliding}" ] && [ -n "${baseline_sliding}" ] && [ "${name}" != "v00_baseline" ]; then
        delta_str=$(awk "BEGIN {d = ${sliding} - ${baseline_sliding}; printf \"%+.4f\", d}")
    fi

    artifact_str="--"
    if [ -n "${artifact_bytes}" ]; then
        artifact_str=$(awk "BEGIN {printf \"%.2f\", ${artifact_bytes} / 1048576.0}")
    fi

    suffix=""
    if [ -n "${notes}" ]; then
        suffix=" (${notes})"
    fi

    printf "%-24s| %-12s| %-11s| %s%s\n" "${name}" "${slide_str}" "${delta_str}" "${artifact_str}" "${suffix}"
done

# Show currently running experiment
if [ -f "${PROGRESS_FILE}" ] && [ -n "${current}" ]; then
    printf "%-24s  (running...)\n" "${current}"
fi

if [ ${has_results} -eq 0 ]; then
    echo "  (no results yet)"
fi

# ── Log tail for current experiment ────────────────────────────────────────

if [ -f "${PROGRESS_FILE}" ] && [ -n "${current}" ]; then
    log_file="${RESULTS_DIR}/${current}.log"
    if [ -f "${log_file}" ]; then
        echo ""
        echo "LOG TAIL (last 5 lines)"
        echo "-----------------------"
        tail -5 "${log_file}"
    fi
fi

echo ""
