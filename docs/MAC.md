# Travailler sur un Mac

> **Rien de ce document n'a été mesuré sur un Mac.** Le dépôt a été écrit et
> vérifié sur un processeur Linux sans GPU. Les durées ci-dessous sont
> *calculées* à partir des FLOPs du modèle et d'ordres de grandeur de puissance
> Apple Silicon, eux-mêmes approximatifs — Apple ne publie pas de chiffre de
> FLOPs crête comparable à celui d'NVIDIA.
>
> Le premier vrai run donnera le débit réel en tokens/s affiché par la boucle.
> C'est ce chiffre-là qu'il faudra reporter ici, en remplaçant les estimations.

Futo détecte le GPU intégré (MPS) tout seul, sans réglage :

```bash
futo info configs/futo-mac.yaml     # affiche votre machine et une estimation
futo train configs/futo-mac.yaml
```

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

**Tout mettre au point.** `futo-tiny` tourne en moins d'une minute. Les 161
tests passent en quelques secondes. Toute la mise au point du code se fait
localement, et seul le run final part sur une machine louée.

## Ce qu'un Mac ne fera pas

Le run complet de `futo-small` : 7,1 EFLOP. Même sur un M3 Ultra à 15 % de MFU,
cela dépasse trois semaines sans interruption. Ce n'est pas raisonnable —
une nuit de mise en veille, une mise à jour système, et c'est perdu. À
comparer aux 5 heures et 20-40 € d'un H100.

La division du travail qui a du sens :

| Étape | Où | Pourquoi |
|---|---|---|
| Tokenizer | Mac | processeur, gratuit, rapide |
| Filtrage et préparation des shards | Mac | processeur, et évite d'envoyer le corpus brut |
| Comparaisons de mélanges (`futo-mac`) | Mac | une nuit par variante, aucun coût |
| `futo-small`, `futo-base` | GPU loué | 20-40 € et 150-250 € respectivement |
| Évaluation, génération, inspection | Mac | quelques secondes, sur le checkpoint rapatrié |

## Les tailles et leurs durées calculées

Durées à 15 % de MFU, valeur prudente pour MPS. À confirmer par la mesure.

| Configuration | Paramètres | Calcul | M2 Max | M3 Max | M3 Ultra |
|---|---|---|---|---|---|
| `futo-tiny` | 1,3 M | — | instantané | instantané | instantané |
| `futo-mac` | 39,3 M | 0,43 EFLOP | 2,4 j | 2,4 j | 1,2 j |
| `futo-small` | 100,7 M | 7,1 EFLOP | 40 j | 39 j | 20 j |

La mémoire n'est pas le facteur limitant. `futo-mac` demande 600 Mio pour les
poids, les gradients et les moments d'Adam ; `futo-small`, 1,5 Gio. Même en
ajoutant les activations, un Mac de 16 Go suffit largement pour les deux. C'est
la *vitesse* qui tranche, pas la mémoire.

## Points techniques propres à MPS

**La précision mixte est essayée, pas supposée.** La prise en charge de
l'autocast sur MPS dépend de la version de PyTorch et du type demandé. Futo
teste la combinaison sur un tenseur minuscule au démarrage : si elle ne passe
pas, il retombe en float32 **en le disant**. Un repli silencieux se paierait en
heures de calcul inexpliquées.

**`torch.compile` : laissez-le désactivé.** Sur MPS il est au mieux inutile, au
pire une source d'erreurs obscures. Le défaut de toutes les configurations est
`compile: false`.

**Pas de `GradScaler`.** Il ne sert qu'au fp16 sur CUDA. Sur MPS, Futo le laisse
désactivé, où il se traverse sans rien faire.

**Le MFU affiché est indicatif.** Il est calculé à partir d'un chiffre de FLOPs
crête estimé, listé dans `FLOPS_CRETE` au début de
[`futo/train.py`](../futo/train.py). Le débit en **tokens/s**, lui, est une
vraie mesure : c'est celui-là qu'il faut regarder et comparer d'un run à
l'autre.

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
