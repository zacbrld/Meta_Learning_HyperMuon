# Meta Learning HyperMuon

Ce depot contient des experiences de meta-learning appliquees a l'optimiseur
Muon. L'objectif est de comparer Muon avec des baselines classiques, puis de
tester des variantes HyperMuon ou certains hyperparametres de Muon sont appris
automatiquement pendant l'entrainement.

Le code entraine des modeles sur CIFAR-10 et logge les performances ainsi que
les trajectoires d'hyperparametres dans des fichiers CSV.

## Idee generale

Muon applique une mise a jour specifique aux parametres matriciels, basee sur :

1. un momentum de type Nesterov ;
2. une normalisation de Frobenius ;
3. une orthogonalisation par iterations de Newton-Schulz ;
4. un scaling RMS cible autour de `0.2` ;
5. une mise a jour avec weight decay.

Les parametres non matriciels, comme les biais et BatchNorm, sont optimises avec
AdamW.

HyperMuon reprend cette structure, mais apprend certains hyperparametres par
hypergradient. Le gradient d'hyperparametre est obtenu avec une approximation
proxy du type :

```text
proxy = - somme_W <grad_W, update_W(theta)>
```

Puis `proxy.backward()` donne un hypergradient sur les hyperparametres
apprenables.

## Optimiseurs disponibles

Le script principal expose sept optimiseurs :

- `sgd` : SGD avec momentum, learning rate schedule cosine.
- `adamw` : AdamW standard, learning rate schedule cosine.
- `hyperadam` : AdamW avec learning rate appris par hypergradient.
- `muon` : Muon avec hyperparametres fixes.
- `hypermuon_l1` : HyperMuon apprend seulement le learning rate `eta`.
- `hypermuon_l2` : HyperMuon apprend `eta` et le momentum `mu`.
- `hypermuon_l3` : HyperMuon apprend `eta`, `mu` et les coefficients
  Newton-Schulz `a`, `b`, `c`.

Les parametrisations utilisees dans HyperMuon sont :

- `lr = exp(lr_raw)` pour garder un learning rate positif ;
- `mu = sigmoid(mu_raw)` pour garder le momentum dans `(0, 1)` ;
- `a`, `b`, `c` sont appris directement au niveau 3.

## Modeles

Deux modeles sont fournis :

- `mlp` : MLP a 3 couches pour CIFAR-10, avec des poids 2D adaptes a Muon.
- `resnet` : ResNet-20 CIFAR-10, avec augmentation de donnees activee.

Le split CIFAR-10 est deterministe :

- 45 000 images train ;
- 5 000 images validation ;
- 10 000 images test.

## Structure du depot

```text
train.py                 boucle principale d'entrainement
plot.py                  generation des figures depuis results/*.csv
run_all.sh               lance plusieurs experiences puis genere les figures
optimizers/
  sgd.py                 wrapper SGD
  adamw.py               wrapper AdamW
  muon.py                Muon fixe
  hyperadam.py           AdamW avec learning rate appris
  hypermuon.py           HyperMuon L1/L2/L3
models/
  mlp.py                 MLP CIFAR-10
  resnet.py              ResNet-20 CIFAR-10
utils/
  data.py                dataloaders CIFAR-10
  logger.py              logger CSV
```

## Installation

```bash
pip install -r requirements.txt
```

PyTorch utilisera CUDA automatiquement si un GPU est disponible.

## Lancer une experience

Exemple avec HyperMuon niveau 3 sur le MLP :

```bash
python train.py --model mlp --optimizer hypermuon_l3 --seed 0
```

Exemple avec Muon fixe sur ResNet-20 :

```bash
python train.py --model resnet --optimizer muon --seed 0
```

Arguments principaux :

```bash
python train.py \
  --model mlp \
  --optimizer hypermuon_l3 \
  --seed 0 \
  --epochs 100 \
  --batch_size 128 \
  --results_dir results/
```

Par defaut, le nombre d'epochs est :

- `100` pour `mlp` ;
- `200` pour `resnet`.

## Lancer la grille d'experiences

```bash
bash run_all.sh
```

Dans son etat actuel, `run_all.sh` lance tous les optimiseurs sur `mlp` avec
`seed=0`, puis appelle `plot.py`. Pour inclure ResNet ou plusieurs seeds, il
faut modifier les variables `MODELS` et `SEEDS` dans ce script.

## Resultats et figures

Chaque run produit un fichier CSV dans `results/`, par exemple :

```text
results/mlp_hypermuon_l3_seed0.csv
```

Les colonnes principales sont :

- pertes train et validation ;
- accuracy validation et test final ;
- valeurs courantes de `lr`, `mu`, `a`, `b`, `c` ;
- normes d'hypergradient ;
- `update_rms`.

Pour generer les figures :

```bash
python plot.py --results_dir results/ --output_dir figures/
```

Les figures produites comparent notamment :

- validation loss vs steps ;
- validation accuracy vs epochs ;
- accuracy test finale ;
- trajectoires des hyperparametres HyperMuon-L3 ;
- normes des hypergradients ;
- RMS des mises a jour Muon/HyperMuon.

## Ce que le projet teste

La question experimentale centrale est :

```text
Peut-on ameliorer ou stabiliser Muon en apprenant automatiquement ses
hyperparametres pendant l'entrainement ?
```

Les niveaux HyperMuon permettent de separer l'effet de chaque famille
d'hyperparametres :

- L1 teste surtout l'adaptation du learning rate ;
- L2 ajoute l'adaptation du momentum ;
- L3 teste aussi si les coefficients Newton-Schulz fixes de Muon peuvent etre
  ajustes par meta-learning.

Les baselines `sgd`, `adamw`, `hyperadam` et `muon` servent a verifier si les
gains viennent vraiment de HyperMuon, ou simplement du choix d'un meilleur
optimiseur ou d'un learning rate adapte.
