# Travailler sur un Mac

> **Ce document repose sur des mesures.** Le dépôt a été écrit sur un processeur
> Linux sans GPU, mais `futo-tiny` et `futo-mac` ont depuis été chronométrés sur
> un **Apple M2 Max** ; les chiffres correspondants sont signalés comme mesurés.
> Ce qui en est extrapolé — `futo-small`, `futo-base`, les autres puces — reste
> un calcul, et est signalé comme tel.
>
> Pour le chiffre exact sur VOTRE machine, sans engager des jours de calcul :
> `futo bench`.

Futo détecte le GPU intégré (MPS) tout seul, sans réglage. Avant toute chose,
mesurez :

```bash
futo bench configs/futo-mac.yaml
```

La commande tire des tokens au hasard — aucun corpus, aucun tokenizer — et
exécute de vraies passes avant et arrière avec le vrai optimiseur, dans la vraie
précision. En une trentaine de secondes, elle répond à la seule question qui
compte avant de lancer un entraînement de plusieurs jours : combien de tokens
par seconde, et donc combien de temps au total. Contrairement à `futo info`, qui
extrapole depuis des FLOPs théoriques, ce chiffre est une mesure.

```bash
futo info configs/futo-mac.yaml    # estimation theorique
futo train configs/futo-mac.yaml
```

## Ce qui a été mesuré sur Apple M2 Max

**`futo-mac`, la configuration qui compte** — 39,3 M de paramètres, contexte
1 024, lots de 8 192 tokens, en **bfloat16** :

| Mesure | Valeur |
|---|---|
| Débit | **9 068 tokens/s** |
| MFU | **19,1 %** |
| Un pas d'optimisation (131 072 tokens) | 14,45 s |
| **Entraînement complet (1,49 G tokens)** | **45,8 h, soit 1,9 jour** |
| Puissance effective | 2,63 TFLOPS |

Deux enseignements. D'abord, **l'autocast bfloat16 fonctionne sur MPS** : la
sonde de démarrage passe, aucun repli sur float32 n'a lieu. Ensuite, les 15 % de
MFU que ce document supposait étaient **conservateurs** — le réel est 19,1 %,
et l'entraînement prend 1,9 jour au lieu des 2,4 annoncés. C'est le bon sens de
l'erreur, mais c'était bien une supposition : d'où la commande `futo bench`.

**`futo-tiny`, pour comparaison** — 1,3 M de paramètres, contexte 256, lots de
2 048 tokens, en float32 :

| Mesure | Apple M2 Max (MPS) | 4 cœurs Linux |
|---|---|---|
| Débit en régime | 57 000 à 60 000 tokens/s | 11 000 à 19 000 tokens/s |
| 400 pas | 15,6 s | 45 à 77 s |
| MFU affiché | 4,0 à 4,2 % | — |
| Suite de tests | 14 s | 18 s |

Le débit brut en tokens/s est six fois plus élevé que sur `futo-mac`, et
pourtant le MFU est cinq fois plus bas : à cette taille, le temps passe dans le
lancement des noyaux, pas dans le calcul. C'est pourquoi un modèle jouet ne dit
jamais rien du débit d'un vrai modèle — et pourquoi `futo bench` mesure la
configuration exacte que vous comptez entraîner, pas une approximation.

---

## Ce pour quoi un Mac est réellement le bon outil

**Entraîner le tokenizer.** C'est du calcul processeur pur, dans une extension
Rust multithread, sans le moindre besoin de GPU. Un Mac à 10-16 cœurs y est
excellent, et il n'y a aucune raison de payer un GPU pour ça.

```bash
futo tokenizer entrainer --corpus 'data/brut/*.jsonl' --vocab 32768 \
                         --octets-max 2000000000
```

**Préparer les données.** Même chose : l'encodage du corpus est processeur, et
la bibliothèque `tokenizers` répartit le travail sur tous les cœurs. C'est
surtout l'argument décisif : préparer localement évite d'envoyer des dizaines
de gigaoctets de corpus brut vers une machine louée à l'heure. On n'y envoie
ensuite que les shards, deux fois plus légers que le texte, et on ne paie pas
le GPU pendant qu'il tokenise.

**Comparer des variantes.** C'est là que `futo-mac` prend son sens : 39 M de
paramètres, 1,5 G de tokens, deux à trois jours sur un M3 Max. Le modèle obtenu
ne vaut pas grand-chose en absolu, mais il permet de trancher entre deux
mélanges de corpus, deux tailles de vocabulaire, deux filtrages — et ces
conclusions-là se transposent le plus souvent à `futo-small`. C'est le bon
endroit pour se tromper, avant de payer un GPU.

**Tout mettre au point.** `futo-tiny` tourne en 16 s sur un M2 Max. Les 190
tests passent en 14 s. Toute la mise au point du code se fait
localement, et seul le run final part sur une machine louée.

## Ce qu'un Mac ne fera pas

Le run complet de `futo-small` : 7,1 EFLOP, soit **31 jours** sur un M2 Max à
l'efficacité mesurée, une quinzaine sur un M3 Ultra. Ce n'est pas raisonnable —
une nuit de mise en veille, une mise à jour système, et c'est perdu. À comparer
aux 5 heures et 20-40 € d'un H100. Et `futo-base` demanderait près d'un an.

En revanche `futo-mac` en 1,9 jour est parfaitement tenable, et c'est bien pour
cela que cette configuration existe.

La division du travail qui a du sens :

| Étape | Où | Pourquoi |
|---|---|---|
| Tokenizer | Mac | processeur, gratuit, rapide |
| Filtrage et préparation des shards | Mac | processeur, et évite d'envoyer le corpus brut |
| Comparaisons de mélanges (`futo-mac`) | Mac | une nuit par variante, aucun coût |
| `futo-small`, `futo-base` | GPU loué | 20-40 € et 150-250 € respectivement |
| Évaluation, génération, inspection | Mac | quelques secondes, sur le checkpoint rapatrié |

## Les tailles et leurs durées

La ligne `futo-mac` sur M2 Max est **mesurée**. Le reste en est extrapolé, à
efficacité égale (19,1 % de MFU) et au prorata des FLOPs crête — donc à prendre
comme un ordre de grandeur. Lancez `futo bench` sur la configuration qui vous
intéresse pour obtenir votre propre chiffre.

| Configuration | Paramètres | Calcul | M2 Max | M3 Max | M3 Ultra |
|---|---|---|---|---|---|
| `futo-tiny` | 1,3 M | — | instantané | instantané | instantané |
| `futo-mac` | 39,3 M | 0,43 EFLOP | **1,9 j (mesuré)** | ~1,8 j | ~0,9 j |
| `futo-small` | 100,7 M | 7,1 EFLOP | ~31 j | ~31 j | ~15 j |
| `futo-base` | 299,4 M | 72 EFLOP | ~317 j | — | — |

La mémoire n'est pas le facteur limitant. `futo-mac` demande 600 Mio pour les
poids, les gradients et les moments d'Adam ; `futo-small`, 1,5 Gio. Même en
ajoutant les activations, un Mac de 16 Go suffit largement pour les deux. C'est
la *vitesse* qui tranche, pas la mémoire.

## Points techniques propres à MPS

**La précision mixte est essayée, pas supposée.** La prise en charge de
l'autocast sur MPS dépend de la version de PyTorch et du type demandé. Futo
teste la combinaison sur un tenseur minuscule au démarrage : si elle ne passe
pas, il retombe en float32 **en le disant**. Un repli silencieux se paierait en
heures de calcul inexpliquées. Constaté sur M2 Max avec PyTorch 2.13 : le
**bfloat16 passe**, la sonde ne déclenche aucun repli.

**`torch.compile` : laissez-le désactivé.** Sur MPS il est au mieux inutile, au
pire une source d'erreurs obscures. Le défaut de toutes les configurations est
`compile: false`.

**Pas de `GradScaler`.** Il ne sert qu'au fp16 sur CUDA. Sur MPS, Futo le laisse
désactivé, où il se traverse sans rien faire.

**Le MFU affiché est indicatif.** Il est calculé à partir d'un chiffre de FLOPs
crête estimé, listé dans `FLOPS_CRETE` au début de
[`futo/train.py`](../futo/train.py), et ce chiffre est une crête **bf16** : un
run en float32 affichera donc un MFU environ deux fois trop bas. Le débit en
**tokens/s**, lui, est une vraie mesure — c'est celui-là qu'il faut regarder et
comparer d'un run à l'autre.

**La reprise après plantage était cassée sur MPS.** Un checkpoint se recharge
avec `map_location=<périphérique>`, ce qui déplaçait aussi l'état du générateur
aléatoire vers le GPU, là où `torch.set_rng_state` exige un tenseur du
processeur. Le défaut touchait MPS **et** CUDA, et passait inaperçu parce que la
suite de tests s'exécute sur processeur. Corrigé, et signalé par un run sur
M2 Max — c'est exactement le genre de faute qu'aucune intégration continue sans
accélérateur ne peut attraper.

**Mémoire unifiée.** Si l'entraînement sature la mémoire, PyTorch expose un
plafond réglable :

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 futo train configs/futo-mac.yaml
```

Avant d'y toucher, préférez baisser `data.batch_size` et augmenter d'autant
`train.grad_accum` : le nombre de tokens par pas reste identique, donc les
hyperparamètres restent valables, et seule l'empreinte mémoire baisse.

```bash
futo train configs/futo-mac.yaml --set data.batch_size=4 --set train.grad_accum=32
```

**Avertissement sur le parallélisme des tokenizers.** Si un message concernant
`TOKENIZERS_PARALLELISM` apparaît pendant la préparation, il est sans
conséquence ici — Futo n'utilise pas de sous-processus de chargement de
données. Pour le faire taire :

```bash
export TOKENIZERS_PARALLELISM=true
```

**Une reprise, toujours.** Une machine de bureau se met en veille, redémarre
pour une mise à jour, tombe en panne de batterie. Sur un run de plusieurs
jours, sauvegardez souvent et reprenez sans y penser :

```bash
futo train configs/futo-mac.yaml --set train.save_every=200
futo train configs/futo-mac.yaml --reprendre auto
```

La reprise est exacte au bit près — c'est vérifié par un test. Aucun pas n'est
rejoué, aucun lot n'est sauté.
