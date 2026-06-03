# HyperMuon — Project Specification
## Tuning Muon Hyperparameters via Automatic Differentiation

> **Contexte :** Ce projet combine deux papers :
> - *Muon is Scalable for LLM Training* — Liu et al., 2025 (arXiv:2502.16982)
> - *Gradient Descent: The Ultimate Optimizer* — Chandra et al., NeurIPS 2022
>
> **Idée centrale :** Appliquer l'auto-différentiation (AD) du paper GD-UO pour tuner
> automatiquement les hyperparamètres de l'optimiseur Muon pendant l'entraînement,
> en particulier les coefficients Newton-Schulz (a, b, c) qui sont normalement fixés à la main.

---

## 1. Dataset & Preprocessing

- **Dataset :** CIFAR-10 (50 000 train, 10 000 test, images 32×32×3, 10 classes)
- **Split :** 45 000 train / 5 000 validation / 10 000 test
- **Normalisation :**
  ```python
  mean = (0.4914, 0.4822, 0.4465)
  std  = (0.2470, 0.2435, 0.2616)
  ```
- **Augmentation (ResNet uniquement) :** RandomHorizontalFlip + RandomCrop(32, padding=4)
- **Seeds :** 3 seeds (0, 1, 2) pour chaque expérience — permet de calculer moyenne ± std

---

## 2. Modèles

### 2.1 MLP (rapide, testable en local)

```
Input  : 3072  (32×32×3 aplati)
Linear : 3072 → 512  + ReLU     poids shape [512, 3072]  ← 2D, Muon natif
Linear : 512  → 256  + ReLU     poids shape [256, 512]   ← 2D, Muon natif
Linear : 256  → 10               poids shape [10,  256]   ← 2D, Muon natif
```

- Pas de BatchNorm
- Epochs : **100**, batch size : **128**
- Les biais (1D) sont traités par AdamW, pas Muon

### 2.2 ResNet-20 (modèle principal, clusters RCP)

- Architecture standard ResNet-20 pour CIFAR-10 (~270k paramètres)
- 3 groupes de blocs résiduels : [16, 32, 64] filtres
- BatchNorm après chaque conv
- Epochs : **200**, batch size : **128**
- Scheduler : cosine annealing (même setup que GD-UO paper)

**Gestion des couches 4D pour Muon :**
Les filtres de convolution sont en 4D `[out_channels, in_channels, kH, kW]`.
Pour appliquer Muon, on reshape en 2D, on orthogonalise, puis on remet en forme :

```python
def apply_muon_to_conv(weight, newton_schulz_fn):
    shape_orig = weight.shape                          # [out, in, kH, kW]
    w_2d = weight.reshape(shape_orig[0], -1)           # [out, in*kH*kW]
    update_2d = newton_schulz_fn(w_2d)                 # Newton-Schulz sur 2D
    return update_2d.reshape(shape_orig)               # retour en 4D

# Muon s'applique à : toutes les couches Conv2d et Linear
# AdamW s'applique à : biais, BatchNorm (weight γ et bias β)
```

---

## 3. Optimiseurs à implémenter

### Vue d'ensemble

| Nom            | Hyperparamètres tunés par AD | Valeurs initiales fixes               |
|----------------|------------------------------|---------------------------------------|
| SGD            | aucun                        | η=0.1, µ=0.9                          |
| AdamW          | aucun                        | η=1e-3, β1=0.9, β2=0.999, λ=0.01     |
| HyperAdam      | η                            | η_init=1e-3, κ=1e-5                   |
| Muon           | aucun                        | µ=0.95, a=3.4445, b=-4.7750, c=2.0315, N=5, η=1e-3, λ=0.1 |
| HyperMuon-L1   | η                            | µ=0.95, a/b/c fixes, κ_η=1e-5        |
| HyperMuon-L2   | η, µ                         | a/b/c fixes, κ_η=1e-5, κ_µ=1e-6     |
| HyperMuon-L3   | η, µ, a, b, c                | κ_η=1e-5, κ_µ=1e-6, κ_abc=1e-7      |

> **HyperAdam** est important : il reproduit le paper GD-UO et valide que le pipeline AD
> fonctionne correctement avant d'être appliqué à Muon.

---

### 3.1 Muon — Algorithme complet

```python
# À chaque step, pour chaque matrice de paramètres W :

# Étape 1 — Momentum Nesterov
g = W.grad.detach()                        # gradient (détaché : pas de dérivées secondes)
M = mu * M_prev + g                        # accumulation momentum

# Étape 2 — Normalisation initiale
X = M / (M.norm(p='fro') + 1e-8)          # normalisation Frobenius

# Étape 3 — Newton-Schulz (N=5 itérations), en float32
X = X.float()
for _ in range(N):
    A = X @ X.T                            # [m, m]
    X = a*X + b*(A @ X) + c*((A @ A) @ X)
# X ≈ UVᵀ (polar factor orthogonal de M)

# Étape 4 — Ajustement RMS pour compatibilité avec AdamW
A_dim, B_dim = W.shape[0], W.shape[1] if W.dim() > 1 else 1
O = 0.2 * X * (max(A_dim, B_dim) ** 0.5)
# RMS(O) = (1/√max(A,B)) * 0.2 * √max(A,B) = 0.2  ✓

# Étape 5 — Mise à jour avec weight decay
W = W.detach() - lr * (O + lambda_ * W.detach())
```

---

### 3.2 HyperMuon — Mécanique AD

Le principe vient du paper GD-UO : on ne `.detach()` pas les hyperparamètres
dans la mise à jour de W, ce qui permet aux hypergradients de remonter automatiquement.

```python
# Les hyperparamètres sont des nn.Parameter dans le graphe de calcul

# Contrainte η > 0 (log-paramétrage)
lr_raw = nn.Parameter(torch.tensor(-6.9))   # log(1e-3)
lr = torch.exp(lr_raw)                       # toujours positif

# Contrainte µ ∈ (0, 1) (sigmoid)
mu_raw = nn.Parameter(torch.tensor(2.944))  # sigmoid(2.944) ≈ 0.95
mu = torch.sigmoid(mu_raw)                   # contraint dans (0,1)

# Coefficients a, b, c — pas de contrainte stricte mais clipping des gradients
a = nn.Parameter(torch.tensor(3.4445))
b = nn.Parameter(torch.tensor(-4.7750))
c = nn.Parameter(torch.tensor(2.0315))

# Dans la update de W :
M = mu * M_prev + g          # mu PAS détaché → hypergradient peut remonter jusqu'à mu_raw
# ... Newton-Schulz avec a, b, c non détachés ...
W = W.detach() - lr * (O + lambda_ * W.detach())
#                ^^^
#                lr PAS détaché → hypergradient peut remonter jusqu'à lr_raw

# Mise à jour des hyperparamètres (après loss.backward()) :
torch.nn.utils.clip_grad_norm_([lr_raw, mu_raw, a, b, c], max_norm=1.0)
lr_raw.data -= kappa_lr  * lr_raw.grad
mu_raw.data -= kappa_mu  * mu_raw.grad
a.data      -= kappa_abc * a.grad
b.data      -= kappa_abc * b.grad
c.data      -= kappa_abc * c.grad

# Remise à zéro des gradients des hyperparamètres
for p in [lr_raw, mu_raw, a, b, c]:
    p.grad = None
```

**Newton-Schulz doit rester différentiable :**
Ne pas utiliser `.detach()` ou `torch.no_grad()` à l'intérieur des itérations
Newton-Schulz pour HyperMuon-L3. Utiliser `.float()` mais garder le graphe intact.

---

## 4. Structure du code

```
hypermuon/
├── train.py                  ← script principal
├── models/
│   ├── __init__.py
│   ├── mlp.py                ← MLP CIFAR-10 (3072→512→256→10)
│   └── resnet.py             ← ResNet-20 CIFAR-10 standard
├── optimizers/
│   ├── __init__.py
│   ├── sgd.py                ← SGD + momentum classique
│   ├── adamw.py              ← AdamW classique
│   ├── hyperadam.py          ← HyperAdam (reproduit GD-UO)
│   ├── muon.py               ← Muon fixe avec Newton-Schulz différentiable
│   └── hypermuon.py          ← HyperMuon L1 / L2 / L3 (level= arg)
├── utils/
│   ├── __init__.py
│   ├── data.py               ← chargement CIFAR-10, splits, transforms
│   └── logger.py             ← écriture CSV step par step
├── results/                  ← CSV générés automatiquement ici
└── plot.py                   ← script séparé, lit CSV et génère figures
```

---

## 5. Logging CSV

**Nom des fichiers :** `results/{model}_{optimizer}_seed{seed}.csv`

Exemples :
```
results/mlp_adamw_seed0.csv
results/mlp_hypermuon_l3_seed1.csv
results/resnet_muon_seed2.csv
```

**Colonnes (toutes les runs, valeurs NaN si non applicable) :**

```
step          ← numéro de batch global (pas epoch)
epoch         ← numéro d'époque
train_loss    ← cross-entropy sur le batch courant
val_loss      ← cross-entropy sur val set (évalué chaque époque)
val_accuracy  ← accuracy sur val set (évalué chaque époque)
test_accuracy ← accuracy sur test set (évalué à la fin seulement)
lr            ← valeur courante de η (exp(lr_raw) si hypermuon)
mu            ← valeur courante de µ (sigmoid si hypermuon, fixe sinon)
a             ← coefficient a Newton-Schulz (fixe ou appris)
b             ← coefficient b Newton-Schulz (fixe ou appris)
c             ← coefficient c Newton-Schulz (fixe ou appris)
hypgrad_lr    ← norme du gradient sur lr_raw (NaN si non applicable)
hypgrad_mu    ← norme du gradient sur mu_raw (NaN si non applicable)
hypgrad_abc   ← norme moyenne des gradients sur a, b, c (NaN si non applicable)
update_rms    ← RMS de la mise à jour appliquée aux poids (moyenne sur toutes les matrices)
```

**Fréquence de log :**
- `train_loss` : chaque step
- Tout le reste : chaque époque (pour ne pas surcharger les CSV)

---

## 6. Interface ligne de commande (train.py)

```bash
python train.py \
  --model       [mlp | resnet]                        \
  --optimizer   [sgd | adamw | hyperadam | muon |
                 hypermuon_l1 | hypermuon_l2 | hypermuon_l3] \
  --seed        [0 | 1 | 2]                           \
  --epochs      [100 pour mlp | 200 pour resnet]      \
  --batch_size  128                                    \
  --results_dir results/
```

**Script pour tout lancer d'un coup :**

```bash
#!/bin/bash
# run_all.sh
MODELS="mlp resnet"
OPTS="sgd adamw hyperadam muon hypermuon_l1 hypermuon_l2 hypermuon_l3"
SEEDS="0 1 2"

for model in $MODELS; do
  for opt in $OPTS; do
    for seed in $SEEDS; do
      echo "Running: $model | $opt | seed $seed"
      python train.py --model $model --optimizer $opt --seed $seed
    done
  done
done
```

---

## 7. Figures (plot.py)

Le script `plot.py` lit tous les CSV dans `results/` et génère les figures suivantes.
Toutes les figures incluent moyenne ± std sur les 3 seeds.

### Figure 1 — Val Loss vs Steps
- Un subplot pour MLP, un pour ResNet
- Toutes les variantes sur le même plot avec légende
- Axe x : steps globaux

### Figure 2 — Val Accuracy vs Epochs
- Même structure que Figure 1

### Figure 3 — Bar chart accuracy finale
- Test accuracy finale (moyenne ± std, 3 seeds)
- Groupé par modèle (MLP / ResNet), une barre par optimizer

### Figure 4 — Trajectoires des hyperparamètres (HyperMuon-L3)
- Subplots : η(t), µ(t), a(t), b(t), c(t) en fonction des epochs
- Lignes horizontales pointillées = valeurs de référence du paper Muon
- Un subplot par hyperparamètre

### Figure 5 — Norme des hypergradients
- `hypgrad_lr`, `hypgrad_mu`, `hypgrad_abc` vs steps pour HyperMuon-L3
- Vérification de la stabilité numérique (doivent rester bornés)

### Figure 6 — RMS des mises à jour
- `update_rms` vs steps pour Muon et HyperMuon
- Ligne de référence à 0.2 (valeur cible AdamW)
- Vérifie que l'ajustement RMS fonctionne

---

## 8. Points d'attention pour l'implémentation

### Newton-Schulz doit tourner en float32
```python
X = X.float()                 # cast avant les itérations
# ... itérations ...
X = X.to(dtype=W.dtype)       # retour au dtype original après
```

### Initialisation de v̂₀ pour HyperAdam
Même problème que mentionné dans GD-UO : `√v̂₀ = 0` cause une division par zéro.
```python
v = torch.full_like(g, fill_value=1e-8)   # initialiser à ε, pas 0
```

### Clipping des hypergradients
```python
torch.nn.utils.clip_grad_norm_([lr_raw, mu_raw, a, b, c], max_norm=1.0)
```

### Reproductibilité
```python
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
```

### Vérification des formes pour Muon
```python
# Muon s'applique seulement aux paramètres matriciels (dim >= 2)
for name, param in model.named_parameters():
    if param.dim() >= 2:
        # → Muon
    else:
        # → AdamW (biais, BatchNorm)
```

---

## 9. Questions de recherche (pour le rapport)

1. **Q1 — L'AD est-elle stable à travers Newton-Schulz ?**
   Les hypergradients restent-ils bornés sur 5 itérations matricielles ?

2. **Q2 — Quel niveau de HyperMuon apporte le plus de gain ?**
   Comparaison L1 vs L2 vs L3 : est-ce que tuner (a,b,c) améliore vraiment ?

3. **Q3 — Les valeurs apprises font-elles sens ?**
   Est-ce que η, µ, a, b, c convergent vers les valeurs "expertes" du paper Muon ?

4. **Q4 — HyperMuon est-il plus robuste aux conditions initiales ?**
   Lancer avec η_init ∈ {1e-4, 1e-3, 1e-2} et comparer la variance finale.

---

## 10. Références

- Liu et al., 2025 — *Muon is Scalable for LLM Training* (arXiv:2502.16982)
- Chandra et al., 2022 — *Gradient Descent: The Ultimate Optimizer* (NeurIPS 2022)
- Jordan et al., 2024 — *Muon: An optimizer for hidden layers* (blog post original)
- Grishina et al., 2025 — *CANS: Chebyshev-optimized Newton-Schulz* (arXiv:2506.10935)
  → Travail connexe le plus proche : optimise (a,b,c) analytiquement, notre approche le fait par AD
