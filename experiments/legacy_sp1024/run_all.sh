#!/bin/bash
set -euo pipefail

# ============================================================================
# Parameter Golf — Master Experiment Runner (1xH200)
# Run from repo root: bash experiments/run_all.sh
# Expected runtime: ~2 hours
# ============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="experiments/results"
CHECKPOINTS_DIR="experiments/checkpoints"
DATA_PATH="data/datasets/fineweb10B_sp1024/"
TOKENIZER_PATH="data/tokenizers/fineweb_1024_bpe.model"
VOCAB_SIZE=1024

TOTAL_TRAINING_EXPERIMENTS=11
EXPERIMENT_INDEX=0
RUN_START_TIME=$(date +%s)

# ── Prereq checks ──────────────────────────────────────────────────────────

if [ ! -d "${DATA_PATH}" ]; then
    echo "ERROR: FineWeb data not found at ${DATA_PATH}"
    echo "Download it first — see README for instructions."
    exit 1
fi

if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "ERROR: Tokenizer not found at ${TOKENIZER_PATH}"
    exit 1
fi

mkdir -p "${RESULTS_DIR}" "${CHECKPOINTS_DIR}"

echo "=== Parameter Golf — Experiment Runner ==="
echo "Start time: $(date)"
echo "Repo root:  ${REPO_ROOT}"
echo ""

# ── Helpers ─────────────────────────────────────────────────────────────────

update_progress() {
    local name="$1"
    local idx="$2"
    local total="$3"
    echo "{\"current\": \"${name}\", \"index\": ${idx}, \"total\": ${total}, \"start_time\": $(date +%s), \"run_start_time\": ${RUN_START_TIME}}" \
        > "${RESULTS_DIR}/progress.json"
}

extract_metrics() {
    local name="$1"
    local log_file="$2"
    local elapsed="$3"

    local val_bpb sliding_bpb val_loss artifact_bytes steps

    # Non-sliding BPB: last "val_bpb:X.XXXX" on a line starting with "step:"
    val_bpb=$(grep -E '^step:.*val_bpb:' "${log_file}" | tail -1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || echo "")

    # Sliding BPB from final_int6_sliding_window_exact
    sliding_bpb=$(grep 'final_int6_sliding_window_exact' "${log_file}" | tail -1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || echo "")

    # Val loss from the same step line
    val_loss=$(grep -E '^step:.*val_loss:' "${log_file}" | tail -1 | sed -E 's/.*val_loss:([0-9.]+).*/\1/' || echo "")

    # Artifact bytes: "Total submission size" line
    artifact_bytes=$(grep 'Total submission size' "${log_file}" | tail -1 | sed -E 's/.*: ([0-9]+) bytes.*/\1/' || echo "")

    # Step count: last "step:NNN/" occurrence
    steps=$(grep -oE 'step:[0-9]+/' "${log_file}" | tail -1 | sed -E 's/step:([0-9]+)\//\1/' || echo "")

    cat > "${RESULTS_DIR}/${name}.json" <<ENDJSON
{
  "experiment": "${name}",
  "val_bpb": ${val_bpb:-null},
  "val_loss": ${val_loss:-null},
  "artifact_bytes": ${artifact_bytes:-null},
  "steps": ${steps:-null},
  "elapsed_s": ${elapsed},
  "sliding_bpb": ${sliding_bpb:-null},
  "notes": ""
}
ENDJSON
}

run_training_experiment() {
    local name="$1"
    EXPERIMENT_INDEX=$((EXPERIMENT_INDEX + 1))
    local dir="experiments/${name}"

    if [ ! -d "${dir}" ]; then
        echo "SKIP: ${dir} does not exist"
        return 0
    fi

    echo ""
    echo "── [${EXPERIMENT_INDEX}/${TOTAL_TRAINING_EXPERIMENTS}] ${name} ──"
    update_progress "${name}" "${EXPERIMENT_INDEX}" "${TOTAL_TRAINING_EXPERIMENTS}"

    local log_file="${RESULTS_DIR}/${name}.log"
    local t_start
    t_start=$(date +%s)

    local status=0
    (
        cd "${dir}"
        RUN_ID="${name}" \
        DATA_PATH="../../${DATA_PATH}" \
        TOKENIZER_PATH="../../${TOKENIZER_PATH}" \
        VOCAB_SIZE="${VOCAB_SIZE}" \
        torchrun --standalone --nproc_per_node=1 train_gpt.py
    ) > "${log_file}" 2>&1 || status=$?
    local t_end
    t_end=$(date +%s)
    local elapsed=$(( t_end - t_start ))

    if [ ${status} -ne 0 ]; then
        echo "FAILED: ${name} (exit ${status}, ${elapsed}s) — see ${log_file}"
        cat > "${RESULTS_DIR}/${name}.json" <<ENDJSON
{
  "experiment": "${name}",
  "val_bpb": null,
  "val_loss": null,
  "artifact_bytes": null,
  "steps": null,
  "elapsed_s": ${elapsed},
  "sliding_bpb": null,
  "notes": "FAILED exit ${status}"
}
ENDJSON
        return 0  # don't kill the batch
    fi

    echo "OK: ${name} (${elapsed}s)"
    extract_metrics "${name}" "${log_file}" "${elapsed}"
}

# ── Phase 0: Train baseline ────────────────────────────────────────────────

echo "=== Phase 0: Baseline training ==="
run_training_experiment "v00_baseline" || true

# Copy baseline artifacts to checkpoints
if [ -f "experiments/v00_baseline/final_model.pt" ]; then
    cp experiments/v00_baseline/final_model.pt "${CHECKPOINTS_DIR}/baseline_raw.pt"
    echo "Copied baseline_raw.pt"
fi
if [ -f "experiments/v00_baseline/final_model.int6.ptz" ]; then
    cp experiments/v00_baseline/final_model.int6.ptz "${CHECKPOINTS_DIR}/baseline_compressed.ptz"
    echo "Copied baseline_compressed.ptz"
fi

# ── Phase 1: Zero-cost eval tests ──────────────────────────────────────────

echo ""
echo "=== Phase 1: Zero-cost eval tests ==="

if [ ! -f "${CHECKPOINTS_DIR}/baseline_compressed.ptz" ]; then
    echo "SKIP Phase 1: baseline checkpoint not found"
else
    for eval_script in eval_stride32 eval_recurrence; do
        echo "── ${eval_script} ──"
        update_progress "${eval_script}" 0 0

        local_log="${RESULTS_DIR}/${eval_script}.log"
        t_start=$(date +%s)

        if uv run "experiments/${eval_script}.py" "${CHECKPOINTS_DIR}/baseline_compressed.ptz" \
                > "${local_log}" 2>&1; then
            t_end=$(date +%s)
            elapsed=$(( t_end - t_start ))
            echo "OK: ${eval_script} (${elapsed}s)"
            # The eval script already wrote ${RESULTS_DIR}/${eval_script}.json — don't overwrite it
        else
            t_end=$(date +%s)
            elapsed=$(( t_end - t_start ))
            echo "FAILED: ${eval_script} (${elapsed}s) — see ${local_log}"
            cat > "${RESULTS_DIR}/${eval_script}.json" <<ENDJSON
{
  "experiment": "${eval_script}",
  "val_bpb": null,
  "val_loss": null,
  "artifact_bytes": null,
  "steps": null,
  "elapsed_s": ${elapsed},
  "sliding_bpb": null,
  "notes": "FAILED"
}
ENDJSON
        fi
    done
fi

# ── Phase 2: Queue A training experiments ───────────────────────────────────

echo ""
echo "=== Phase 2: Queue A training experiments ==="

for exp in v01_xsa_all v02_rebalance_8kv v03_no_resid_mix_dec v04_trigram_conv \
           v05_no_value_embed v08_engram_lite v09_dca_scalar v10_dca_full; do
    run_training_experiment "${exp}" || true
done

# ── Phase 3: Quantization frontier sweep ───────────────────────────────────

echo ""
echo "=== Phase 3: Quantization frontier sweep ==="

if [ ! -f "${CHECKPOINTS_DIR}/baseline_raw.pt" ]; then
    echo "SKIP Phase 3: baseline_raw.pt not found"
else
    update_progress "quant_frontier" 0 0
    quant_log="${RESULTS_DIR}/quant_frontier.log"
    t_start=$(date +%s)

    if uv run experiments/quant_frontier.py "${CHECKPOINTS_DIR}/baseline_raw.pt" \
            > "${quant_log}" 2>&1; then
        t_end=$(date +%s)
        echo "OK: quant_frontier ($((t_end - t_start))s)"
    else
        t_end=$(date +%s)
        echo "FAILED: quant_frontier ($((t_end - t_start))s) — see ${quant_log}"
    fi
fi

# ── Phase 4: Queue B training experiments ───────────────────────────────────

echo ""
echo "=== Phase 4: Queue B training experiments ==="

for exp in v06_12L_int5mlp v07_ushaped_quant; do
    run_training_experiment "${exp}" || true
done

# ── Final: Results table ────────────────────────────────────────────────────

echo ""
echo "=== Generating results table ==="

# Read baseline sliding BPB for delta computation
BASELINE_SLIDING=""
if [ -f "${RESULTS_DIR}/v00_baseline.json" ]; then
    BASELINE_SLIDING=$(sed -n 's/.*"sliding_bpb": *\([0-9.]*\).*/\1/p' "${RESULTS_DIR}/v00_baseline.json")
fi

print_results_table() {
    local order=(
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

    printf "\n=== FINAL RESULTS (1xH200, 600s wallclock) ===\n"
    printf "%-24s| %-9s| %-9s| %-11s| %-12s| %s\n" \
        "Experiment" "val_bpb" "sliding" "D baseline" "artifact_mb" "notes"
    printf "========================|==========|==========|============|=============|======\n"

    local winners=""

    for name in "${order[@]}"; do
        local json_file="${RESULTS_DIR}/${name}.json"
        if [ ! -f "${json_file}" ]; then
            continue
        fi

        local val_bpb sliding artifact_bytes notes
        val_bpb=$(sed -n 's/.*"val_bpb": *\([0-9.]*\).*/\1/p' "${json_file}")
        sliding=$(sed -n 's/.*"sliding_bpb": *\([0-9.]*\).*/\1/p' "${json_file}")
        artifact_bytes=$(sed -n 's/.*"artifact_bytes": *\([0-9]*\).*/\1/p' "${json_file}")
        notes=$(sed -n 's/.*"notes": *"\([^"]*\)".*/\1/p' "${json_file}")

        # Format values
        local val_str="--"
        if [ -n "${val_bpb}" ]; then
            val_str="${val_bpb}"
        fi

        local slide_str="--"
        if [ -n "${sliding}" ]; then
            slide_str="${sliding}"
        fi

        local delta_str="--"
        if [ -n "${sliding}" ] && [ -n "${BASELINE_SLIDING}" ] && [ "${name}" != "v00_baseline" ]; then
            delta_str=$(awk "BEGIN {d = ${sliding} - ${BASELINE_SLIDING}; printf \"%+.4f\", d}")
        fi

        local artifact_str="--"
        if [ -n "${artifact_bytes}" ]; then
            artifact_str=$(awk "BEGIN {printf \"%.2f\", ${artifact_bytes} / 1048576.0}")
        fi

        printf "%-24s| %-9s| %-9s| %-11s| %-12s| %s\n" \
            "${name}" "${val_str}" "${slide_str}" "${delta_str}" "${artifact_str}" "${notes}"

        # Track winners
        if [ -n "${sliding}" ] && [ -n "${BASELINE_SLIDING}" ] && [ "${name}" != "v00_baseline" ]; then
            local is_better
            is_better=$(awk "BEGIN {print (${sliding} < ${BASELINE_SLIDING}) ? 1 : 0}")
            if [ "${is_better}" = "1" ]; then
                winners="${winners}  ${name} (${slide_str}, ${delta_str})\n"
            fi
        fi
    done

    echo ""
    echo "=== WINNERS (sliding BPB better than baseline) ==="
    if [ -n "${winners}" ]; then
        printf "%b" "${winners}"
    else
        echo "  (none)"
    fi
    echo ""
}

# Print to stdout and save to file
print_results_table | tee "${RESULTS_DIR}/FINAL_RESULTS.txt"

# Clear progress
rm -f "${RESULTS_DIR}/progress.json"

echo "Done at $(date)"
echo "Results saved to ${RESULTS_DIR}/FINAL_RESULTS.txt"
