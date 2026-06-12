#!/bin/bash
echo "Syncing code to Izar..."
rsync -avz -e "ssh -o StrictHostKeyChecking=no" --exclude '.git' --exclude 'datasets' /Users/thomaschetaille/Documents/Meta_Learning_HyperMuon/external/llm-baselines/ chetaill@izar.epfl.ch:/home/chetaill/muon/external/llm-baselines/
rsync -avz -e "ssh -o StrictHostKeyChecking=no" /Users/thomaschetaille/Documents/Meta_Learning_HyperMuon/run_fineweb_layerwise_izar.slurm chetaill@izar.epfl.ch:/home/chetaill/muon/
rsync -avz -e "ssh -o StrictHostKeyChecking=no" /Users/thomaschetaille/Documents/Meta_Learning_HyperMuon/run_wikitext_51m_layerwise_izar.slurm chetaill@izar.epfl.ch:/home/chetaill/muon/
rsync -avz -e "ssh -o StrictHostKeyChecking=no" /Users/thomaschetaille/Documents/Meta_Learning_HyperMuon/run_wikitext_layerwise_izar.slurm chetaill@izar.epfl.ch:/home/chetaill/muon/

SCRIPT="${SCRIPT:-run_fineweb_layerwise_izar.slurm}"
case "${SCRIPT}" in
  run_fineweb_layerwise_izar.slurm|run_wikitext_51m_layerwise_izar.slurm|run_wikitext_layerwise_izar.slurm) ;;
  *) echo "Unknown SCRIPT=${SCRIPT}"; exit 1 ;;
esac

echo "Submitting SLURM job: ${SCRIPT}"
ssh -o StrictHostKeyChecking=no chetaill@izar.epfl.ch "cd /home/chetaill/muon && sbatch ${SCRIPT}"
