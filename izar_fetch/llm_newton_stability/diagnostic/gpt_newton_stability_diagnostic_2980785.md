# GPT Newton-Muon Stability Diagnostic

Jobs: `2980785_[0-2]`, all `COMPLETED` with empty `.err` files.

| tag | final val loss | best val loss | final acc | mean iter dt | clipped frac mean | max precond ratio |
|---|---:|---:|---:|---:|---:|---:|
| lr0p005_ridge0p5_clip3 | 4.050 | 4.050 | 0.344421 | 0.615s | 0.311 | 233.5 |
| lr0p004_ridge1p0_clip3 | 4.080 | 4.080 | 0.340409 | 0.562s | 0.272 | 1040.0 |
| lr0p004_ridge0p5_clip3 | 4.097 | 4.097 | 0.337765 | 0.653s | 0.368 | 157.3 |

Reference previous run: AdamW final val loss 4.472; Muon final val loss 4.096; old Newton-Muon best finite 4.159 at iter 1400 then NaN.

Interpretation: lowering Newton-Muon LR and adding preconditioner clipping fixed the NaN failure. The best stable setting here is `lr0p005_ridge0p5_clip3`, which finishes at `4.050`, slightly better than the Muon baseline on this short 2000-step WikiText run.
