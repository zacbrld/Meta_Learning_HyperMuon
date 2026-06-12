#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./izar_fetch_and_summarize.sh <job_id> [remote_dir_name]"
    echo "Example: ./izar_fetch_and_summarize.sh 2988587 results_llm_gpt_gduo_meta"
    exit 1
fi

JOB_ID=$1
DIR_NAME=$2

# Fetch results and logs
./utils/fetch_izar.sh "${JOB_ID}" "${DIR_NAME}"

# Parse logs and summarize metrics
python utils/summarize_runs.py "${JOB_ID}"
