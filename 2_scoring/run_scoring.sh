#!/bin/bash
# ---------------------------------------------------------------------------
# PopResume — resume scoring runner
#
# Replaces the four per-provider scripts with one configurable entry point.
# Everything below the CONFIG block is fixed to the paper's setup:
#   * all five occupations
#   * resumes rendered WITHOUT a skills section  (--resume_file_key no_skills)
#   * demographics-augmented variants excluded   (--resume_file_ignore_key with_demographics)
#   * job description version v2                 (2_scoring/job_info_v2/)
#
# Usage:
#   export DATA_ROOT=/your/CausalFair
#   ./run_scoring.sh                    # uses the CONFIG below
#   MODE=vlm ./run_scoring.sh           # or override from the environment
#   MODE=llm PROVIDER=openai MODELS="gpt-4o-mini" ./run_scoring.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# ============================== CONFIG =====================================
MODE="${MODE:-llm}"            # llm | vlm
PROVIDER="${PROVIDER:-hf}"     # hf | openai | gemini      (MODE=llm only)
GPUS="${GPUS:-0 1 2 3}"        # GPUs to fan out across
BATCH_SIZE="${BATCH_SIZE:-32}" # local HF models only

# Models to score with. Paper configuration:
#   MODE=llm PROVIDER=hf      -> llama-3.1-8b-instruct, mistral-7b-instruct-v0.2
#   MODE=llm PROVIDER=openai  -> gpt-4o-mini
#   MODE=llm PROVIDER=gemini  -> gemini-2.5-flash-lite
#   MODE=vlm                  -> qwen2.5-vl-7b-instruct, internvl2-8b, gpt-4o, gemini-2.5-flash-lite
MODELS="${MODELS:-llama-3.1-8b-instruct mistral-7b-instruct-v0.2}"
# ===========================================================================

: "${DATA_ROOT:?set DATA_ROOT to your CausalFair checkout, e.g. export DATA_ROOT=/data/CausalFair}"

BASE_DIR="${DATA_ROOT}/Resume/job-distribution"
PREFIX="processed_job_data_0102_with_exp_pred_with_names_with_resumes_datafull"
JOB_VER="v2"

JOBS=(
    construction_laborers
    registered_nurses
    software_developers
    elementary_middle_school_teachers
    accountants_auditors
)

case "${MODE}" in
    llm) SUFFIXES=(resumes_txt) ;;
    vlm) SUFFIXES=(resumes_jpg resumes_jpg_no_images) ;;
    *)   echo "MODE must be 'llm' or 'vlm' (got '${MODE}')" >&2; exit 2 ;;
esac

read -r -a GPU_ARR <<< "${GPUS}"
read -r -a MODEL_ARR <<< "${MODELS}"
NUM_GPUS=${#GPU_ARR[@]}
FAILED=0

run_one() {
    local folder="$1" gpu="$2" model="$3"
    if [[ "${MODE}" == "vlm" ]]; then
        CUDA_VISIBLE_DEVICES="${gpu}" python score_resumes_vlm_score.py \
            --folder_dir "${folder}" \
            --vlm_model_name "${model}" \
            --job_info_version "${JOB_VER}" \
            --resume_file_key "no_skills" \
            --resume_file_ignore_key "with_demographics"
    elif [[ "${PROVIDER}" == "hf" ]]; then
        CUDA_VISIBLE_DEVICES="${gpu}" python score_resumes_llm_score.py \
            --folder_dir "${folder}" \
            --provider hf \
            --llm_model_name "${model}" \
            --batch_size "${BATCH_SIZE}" \
            --job_info_version "${JOB_VER}" \
            --resume_file_key "no_skills" \
            --resume_file_ignore_key "with_demographics"
    else
        # API providers: no local GPU, no batching
        python score_resumes_llm_score.py \
            --folder_dir "${folder}" \
            --provider "${PROVIDER}" \
            --llm_model_name "${model}" \
            --job_info_version "${JOB_VER}" \
            --resume_file_key "no_skills" \
            --resume_file_ignore_key "with_demographics"
    fi
}

for model in "${MODEL_ARR[@]}"; do
    TASKS=()
    for suffix in "${SUFFIXES[@]}"; do
        for job in "${JOBS[@]}"; do
            TASKS+=("${BASE_DIR}/${PREFIX}_${job}_${suffix}")
        done
    done

    echo "=== ${MODE}/${PROVIDER} | model: ${model} | tasks: ${#TASKS[@]} | gpus: ${GPUS} ==="

    for ((i = 0; i < ${#TASKS[@]}; i += NUM_GPUS)); do
        PIDS=()
        for ((j = 0; j < NUM_GPUS && i + j < ${#TASKS[@]}; j++)); do
            folder="${TASKS[i + j]}"
            echo "[gpu ${GPU_ARR[j]}] [$((i + j + 1))/${#TASKS[@]}] ${folder##*/}"
            run_one "${folder}" "${GPU_ARR[j]}" "${model}" &
            PIDS+=($!)
        done
        for pid in "${PIDS[@]}"; do
            wait "${pid}" || FAILED=$((FAILED + 1))   # plain assignment: always exit 0 under `set -e`
        done
    done

    echo "=== done: ${model} ==="
done

if ((FAILED > 0)); then
    echo "WARNING: ${FAILED} task(s) failed." >&2
    exit 1
fi
echo "All scoring tasks completed."
