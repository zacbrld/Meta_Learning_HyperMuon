# Rapport: Meta-Learning HyperMuon

Ce document sert de squelette pour le rapport. L'idee centrale est de partir du
meta-learning d'optimiseurs, puis d'expliquer comment on passe d'un apprentissage
du temps d'optimisation, c'est-a-dire learning rate et momentum, vers un
apprentissage de la geometrie, c'est-a-dire des directions et preconditionnements
qui modifient la courbure vue par Muon.

## 1. Motivation

Les optimiseurs modernes contiennent beaucoup de choix fixes: learning rate,
momentum, normalisation, preconditionnement, coefficients de projection, ridge,
etc. En pratique, ces hyperparametres controlent deux choses differentes:

- la dynamique temporelle: combien avancer, avec quelle inertie, et selon quel
  planning;
- la geometrie locale: dans quelle direction avancer quand le gradient brut est
  mal conditionne.

Notre objectif est de tester si une partie de ces choix peut etre apprise pendant
l'entrainement, au lieu d'etre seulement reglee par grid search.

## 2. Papier 1: Gradient Descent, The Ultimate Optimizer

Reference: `/Users/thomaschetaille/Downloads/1909.13371v2-2.pdf`

Ce papier defend l'idee que les hyperparametres d'un optimiseur peuvent eux-memes
etre optimises par descente de gradient. Le cas simple est le learning rate:
apres une etape de poids, on peut deriver la perte suivante par rapport au
learning rate precedent. Cela donne un signal d'hypergradient.

Dans notre projet, ce papier justifie le passage de schedules fixes a des
hyperparametres appris en ligne:

- learning rate global;
- learning rate par couche;
- momentum par couche;
- plus tard, force de preconditionnement par bucket.

Le message important pour le rapport: on n'apprend pas directement les poids du
modele plus vite par magie; on apprend des parametres de l'optimiseur qui
controlent comment les poids sont mis a jour.

## 3. Papier 2: The Newton-Muon Optimizer

Reference: `/Users/thomaschetaille/Downloads/2604.01472v1-2.pdf`

Muon projette les updates matricielles avec une operation proche du signe
matriciel. Newton-Muon propose une interpretation plus geometrique: l'update peut
etre vue comme une approximation de type Newton sur un surrogate quadratique, avec
une courbure de sortie et une covariance d'entree.

Le point cle pour notre travail:

- Muon donne une bonne direction de base;
- Newton-Muon essaie d'ajouter de la courbure via un preconditionnement;
- cette courbure peut etre utile, mais elle peut aussi etre instable si l'inverse
  de covariance est mal conditionne.

Notre question experimentale devient donc:

> Peut-on apprendre quand et ou utiliser cette information geometrique, au lieu de
> l'appliquer uniformement a toutes les couches?

## 4. Notre hypothese

On separe le probleme en deux axes.

### Axe A: apprendre le temps

Ici on apprend des hyperparametres de dynamique:

- learning rate;
- momentum;
- version layerwise pour permettre a chaque couche d'avoir son rythme.

Resultat principal: cet axe marche. Le meilleur run Muon-GDUO layerwise sur GPT
WikiText atteint une perte nettement meilleure que les baselines courtes, ce qui
montre que le meta-learning apprend bien une dynamique d'optimisation utile.

### Axe B: apprendre la geometrie et la courbure

Ici on ne veut plus seulement apprendre "combien avancer", mais "dans quelle
geometrie avancer". On teste donc des variantes ou le gradient est transforme
avant ou autour de Muon:

```text
geo_grad = grad + s_bucket * (precond_grad - grad)
update   = Muon(momentum(geo_grad))
```

Avec cette forme, Muon reste le socle. Le preconditionneur ne remplace pas Muon:
il propose une correction de courbure apprise, controlee par bucket.

Les buckets minimaux sont:

- `attn_qkv`
- `attn_proj`
- `mlp_fc`
- `mlp_proj`
- embeddings / head separes ou exclus selon le run

## 5. Methodes testees

### Baselines fixes

- AdamW
- Muon
- Newton-Muon

### Meta-learning de dynamique

- AdamW-GDUO
- Muon-GDUO
- Muon-GDUO layerwise LR + momentum

### Meta-learning de geometrie

- Adam/Muon residual gate: correction Adam autour de Muon;
- Muon/Newton-Muon residual gate: correction Newton autour de Muon;
- Muon/AdaGrad-EMA/Muon: preconditionnement diagonal EMA avant Muon;
- SOAP-lite/Muon: preconditionnement matriciel simplifie avant Muon.

## 6. Resultats actuels

### CIFAR-10: reproduction Newton-Muon

Modele: ResMLP 32 couches, width 512.

| Optimiseur | Final acc | Best acc |
|---|---:|---:|
| AdamW | 60.51% | 60.98% |
| Muon | 66.24% | 66.39% |
| Newton-Muon | 67.33% | 67.49% |

Conclusion: le signal du papier Newton-Muon est reproduit sur CIFAR. Newton-Muon
bat Muon dans ce cadre controle.

### WikiText GPT 51M: baselines courtes

Runs a 500 iterations.

| Run | Optimiseur | Val loss | Accuracy |
|---|---|---:|---:|
| `3007222_0` | AdamW | 5.203 | 0.2355 |
| `3007222_1` | Muon | 4.461 | 0.3030 |
| `3007222_2` | Newton-Muon ridge 0.5 | 4.793 | 0.2652 |
| `3007222_3` | Newton-Muon ridge 1.0 | 4.774 | 0.2675 |

Conclusion: sur GPT WikiText, Newton-Muon brut ne bat pas Muon. C'est justement
ce qui motive l'apprentissage de la geometrie au lieu d'une application uniforme
du preconditionnement Newton.

### WikiText GPT 51M: LR + momentum layerwise

Run principal:

| Run | Optimiseur | Iterations | Best val loss | Final val loss | Accuracy |
|---|---|---:|---:|---:|---:|
| `3005857_1` | Muon-GDUO LR+momentum layerwise | 2000 | 3.468 | 3.477 | 0.3932 |

Conclusion: apprendre le temps et le momentum par couche est utile. C'est notre
resultat positif le plus clair.

### WikiText GPT 51M: apprentissage de geometrie a 500 iterations

| Run | Variante | Val loss | Accuracy | Temps / iter | Tok/s |
|---|---|---:|---:|---:|---:|
| `3027436_0` | Adam/Muon residual alpha 0.05 | 4.470 | 0.3004 | 2.21s | 22.2k |
| `3027436_1` | Muon/Newton residual alpha 0.05 | 4.804 | 0.2644 | 2.53s | 19.4k |
| `3028508_0` | Adam/Muon bucket corrige | 4.464 | 0.3009 | 2.22s | 22.2k |
| `3028508_1` | Muon/Newton bucket corrige | 4.809 | 0.2639 | 2.53s | 19.4k |
| `3028532_0` | Adam/Muon gate plus fort | 4.485 | 0.2988 | 2.22s | - |
| `3028534_0` | Muon/AdaGrad-EMA/Muon | 4.435 | 0.3054 | 2.13s | 23.1k |
| `3028534_1` | SOAP-lite/Muon | 4.420 | 0.3070 | 2.30s | 21.4k |

Interpretation:

- Adam/Muon ne gagne pas clairement contre Muon, meme avec gate plus fort.
- Muon/Newton-Muon est mauvais dans ce setup: clipping fort et directions souvent
  redondantes ou dangereuses.
- AdaGrad-EMA/Muon et SOAP-lite/Muon donnent les meilleurs signaux courts.
- AdaGrad-EMA/Muon est aussi plus rapide que Muon dans nos runs courts.
- SOAP-lite/Muon donne la meilleure perte a 500 iterations, mais avec un cout
  temps un peu plus eleve.

## 7. Ce qu'on apprend sur la courbure

Newton-Muon brut essaie d'apprendre/utiliser une courbure plus riche, mais notre
diagnostic montre deux problemes:

- certains buckets clippent beaucoup, notamment `mlp_proj` et `attn_proj`;
- quand `cos(muon, newton)` est tres proche de 1, Newton n'apporte surtout qu'un
  changement d'echelle, pas une vraie nouvelle direction.

La conclusion pragmatique est:

> La courbure utile doit etre apprise par bucket et injectee avant Muon, avec une
> force initiale faible.

Les variantes AdaGrad-EMA et SOAP-lite suivent mieux ce principe:

- elles proposent une correction geometrique stable;
- elles laissent Muon faire la projection finale;
- elles apprennent une force `s_bucket` plutot que de choisir brutalement entre
  Muon et un autre optimiseur.

## 8. Runs en cours / prochains resultats

Les runs 2000 iterations lances pour confirmer les meilleurs signaux courts sont:

| Job | Variante | Iterations |
|---|---|---:|
| `3028637_0` | Muon/AdaGrad-EMA/Muon | 2000 |
| `3028637_1` | SOAP-lite/Muon | 2000 |

Ces runs doivent etre compares a:

- Muon baseline 500/2000;
- Muon-GDUO LR+momentum layerwise;
- Newton-Muon brut;
- wall-clock time, pas seulement nombre d'iterations.

## 9. Figures disponibles

Figures utiles pour le rapport:

```text
figures/paper_like/cifar10_resmlp32_accuracy_paper.png
figures/paper_like/figure1_cifar_gpt_paper.png
figures/paper_like/figure1_four_panel_paper.png
figures/paper_like/gpt_wikitext_val_loss_paper.png
figures/paper_like/gpt_wikitext_val_perplexity_paper.png
figures/paper_like/gpt_wikitext_newton_stability_sweep.png

figures/meta_learning/gpt_wikitext_baselines_vs_meta.png
figures/meta_learning/gpt_wikitext_accuracy_baselines_vs_meta.png
figures/meta_learning/gpt_wikitext_perplexity_baselines_vs_meta.png
figures/meta_learning/gpt_wikitext_meta_loss_vs_time.png
figures/meta_learning/gpt_wikitext_meta_throughput.png
```

## 10. Plan conseille pour le rapport

1. Introduction: pourquoi apprendre un optimiseur?
2. Meta-learning et hypergradients: resume du papier GD-UO.
3. Muon et Newton-Muon: resume du papier Newton-Muon.
4. Notre question: apprendre le temps puis apprendre la geometrie.
5. Implementation:
   - GD-UO LR/momentum layerwise;
   - gates residuels;
   - preconditionnement par bucket avant Muon.
6. Resultats:
   - CIFAR reproduction;
   - GPT WikiText baselines;
   - LR+momentum layerwise;
   - geometrie/courbure.
7. Discussion:
   - ce qui marche;
   - pourquoi Newton brut est fragile;
   - pourquoi AdaGrad-EMA et SOAP-lite sont prometteurs.
8. Conclusion et travaux futurs.

## 11. Travaux futurs

- Confirmer AdaGrad-EMA/Muon et SOAP-lite/Muon a 2000 iterations.
- Ajouter plusieurs seeds.
- Comparer perte en fonction du temps mur, pas seulement des steps.
- Ablater `s_bucket`, `beta`, `eps`, et la decomposition par bucket.
- Filtrer automatiquement les corrections si `cos(muon, precond) > 0.9` ou si
  le clipping est trop fort.
- Tester une version Newton plus stable: ridge adaptatif par bucket, disable
  Newton sur `mlp_proj` et `attn_proj`, ou diagonal/block inverse au lieu d'un
  inverse dense fragile.
