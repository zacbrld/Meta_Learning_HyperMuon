import glob
import os
import re
import sys

def summarize(job_id):
    results = []
    # Find all out files for this job ID
    for filepath in glob.glob(f'izar_fetch/logs/*{job_id}*.out'):
        tag = 'Unknown'
        val_loss = -1
        val_pp = -1
        val_acc = -1
        iter_dt = -1
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'tag=' in line:
                    tag = line.split('tag=')[1].strip()
                if '>Eval: Iter=' in line:
                    m = re.search(r'val_loss=([\d\.]+)\s+val_pp=([\d\.]+)\s+val_acc=([\d\.]+)', line)
                    if m:
                        val_loss = float(m.group(1))
                        val_pp = float(m.group(2))
                        val_acc = float(m.group(3))
                if 'Train: Iter=' in line:
                    m = re.search(r'iter_dt=([\d\.eE+-]+)s', line)
                    if m:
                        iter_dt = float(m.group(1))
                        
        if val_loss != -1:
            results.append({
                'Config': tag,
                'Val Loss': val_loss,
                'Val PPL': val_pp,
                'Val Acc': val_acc,
                'Iter dt': iter_dt
            })

    if not results:
        print(f"No completed results found for job ID {job_id}.")
        return

    print(f"\n{'Config':<35} | {'Val Loss':<10} | {'Val PPL':<10} | {'Val Acc':<10} | {'Iter dt (s)':<10}")
    print('-' * 85)
    for r in sorted(results, key=lambda x: x['Val Loss']):
        print(f"{r['Config']:<35} | {r['Val Loss']:<10.4f} | {r['Val PPL']:<10.4f} | {r['Val Acc']:<10.4f} | {r['Iter dt']:<10.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils/summarize_runs.py <job_id>")
        sys.exit(1)
    summarize(sys.argv[1])
