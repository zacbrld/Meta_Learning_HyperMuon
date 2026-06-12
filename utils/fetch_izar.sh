#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./utils/fetch_izar.sh <job_id> [remote_dir_name]"
    exit 1
fi

JOB_ID=$1
DIR_NAME=${2:-results_llm_gpt_gduo_meta}
DEST_DIR="izar_fetch/${DIR_NAME}"

echo "=========================================================="
echo " Fetching results for Job ID: ${JOB_ID}"
echo " Remote Directory: /home/chetaill/muon/${DIR_NAME}/"
echo "=========================================================="

mkdir -p "${DEST_DIR}"
mkdir -p izar_fetch/logs

# Fetch results directory
rsync -az -e "ssh -o StrictHostKeyChecking=no" chetaill@izar.epfl.ch:/home/chetaill/muon/${DIR_NAME}/ "${DEST_DIR}/"

# Fetch logs corresponding to this job ID
rsync -az -e "ssh -o StrictHostKeyChecking=no" "chetaill@izar.epfl.ch:/home/chetaill/muon/logs/*${JOB_ID}*" izar_fetch/logs/

echo "Fetch complete."
