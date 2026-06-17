#! /bin/bash

AV_HUBERT=$(dirname "$(dirname "$(readlink -fm "$0")")")
ROOT=$(dirname "${AV_HUBERT}")
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"

MODEL_PATH=/workspace/code/visg_large_best.pt
RESULTS_PATH=$(dirname "$(readlink -fm "$0")")/results/speech_snr-10

NUM_RUNS=3
wer_sum=0

for RUN in $(seq 1 ${NUM_RUNS}); do
    RUN_PATH="${RESULTS_PATH}/run${RUN}"
    mkdir -p "${RUN_PATH}"
    echo ">>> Run ${RUN}/${NUM_RUNS}"

    python -B ${AV_HUBERT}/infer_s2s.py \
        --config-dir ${AV_HUBERT}/conf \
        --config-name s2s_decode \
        dataset.gen_subset=test \
        common_eval.path=${MODEL_PATH} \
        common_eval.results_path=${RUN_PATH} \
        override.modalities=['audio','video'] \
        common.user_dir=${AV_HUBERT} \
        override.data=/DB/lrs3/433h_data \
        override.label_dir=/DB/lrs3/433h_data \
        override.noise_prob=1.0 \
        override.noise_snr=-10 \
        +override.noise_wav=/DB/lrs3/noise/speech \
        distributed_training.distributed_world_size=1 2>&1 | tee "${RUN_PATH}/log.txt"

    WER=$(grep -oP "WER:\s*\K[0-9.]+" "${RUN_PATH}/log.txt" | tail -1)
    echo ">>> Run ${RUN} WER: ${WER}%"
    wer_sum=$(awk -v s="${wer_sum}" -v w="${WER}" 'BEGIN { printf "%.6f", s + w }')
done

avg_wer=$(awk -v s="${wer_sum}" -v n="${NUM_RUNS}" 'BEGIN { printf "%.4f", s / n }')
echo ""
echo "=============================="
echo "  Runs: ${NUM_RUNS}"
echo "  Average WER: ${avg_wer}%"
echo "=============================="
