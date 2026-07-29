# Futo — un modèle de langue français entraîné de zéro

Le nom vient de « futé ».

Futo est la chaîne complète qui va d'un corpus de textes français à un modèle
qui écrit du français : le tokenizer, la préparation des données, le modèle,
l'entraînement, l'évaluation, la génération. Tout est écrit à la main, en
PyTorch, sans framework d'entraînement, avec quatre dépendances en tout. On doit
pouvoir lire le moteur en une soirée et le modifier le lendemain.

**Ce que c'est.** Un dépôt qui marche tout de suite : on clone, on lance, un
modèle s'entraîne sur le processeur en une minute, et la chaîne est la même que
celle qui servira sur un GPU loué.

**Ce que ce n'est pas.** Un modèle entraîné. Le dépôt fournit la machine et un
corpus d'exemple de 142 Ko qui ne sert qu'aux tests. Rassembler les milliards de
tokens de français nécessaires est un travail à part, décrit dans
[docs/DONNEES.md](docs/DONNEES.md) — et c'est le vrai obstacle du projet, bien
plus que le calcul.

---

## Essayer en cinq minutes

```bash
git clone https://github.com/kirea-sas/futo && cd futo

# Un environnement isole. Obligatoire avec Homebrew et plusieurs distributions
# Linux, qui refusent desormais toute installation dans le Python du systeme
# (PEP 668). Cela evite aussi de melanger les versions de PyTorch.
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Un tokenizer francais, entraine sur le corpus fourni dans le depot
futo tokenizer entrainer --corpus 'data/echantillon/*.txt' --vocab 4096

# 2. Le corpus, encode en shards binaires
futo data preparer --corpus 'data/echantillon/*.txt' --fraction-val 0.05 \
                   --tokens-par-shard 30000

# 3. Un modele de 1,3 M de parametres, sur processeur
futo train configs/futo-tiny.yaml --set train.max_steps=400

# 4. Voir ce que le modele a appris
futo eval sorties/tiny/meilleur.pt
futo generer sorties/tiny/dernier.pt --amorce "Le vieux moulin" --max-tokens 60
```

Il faut **Python 3.10 ou plus récent** (`python3 --version`). Sur macOS, la
commande s'appelle `python3`, jamais `python`, et `pip` n'existe qu'une fois
l'environnement activé — c'est la raison du `source .venv/bin/activate`
ci-dessus. Pensez à le réactiver à chaque nouveau terminal.

Passé l'installation, aucune de ces commandes ne touche au réseau.
L'entraînement prend 45 à 80 secondes sur quatre cœurs, selon la charge de la
machine.

Le texte produit sera du charabia — 1,3 million de paramètres entraînés sur
37 000 tokens, on n'attend rien d'autre. Ce que ces quatre commandes prouvent,
c'est que la chaîne est complète et qu'elle tourne. La suite n'est qu'une
affaire d'échelle.

---

## Les cinq tailles

| Configuration | Paramètres | Couches | d_model | Contexte | Budget visé | Calcul | Où l'entraîner |
|---|---|---|---|---|---|---|---|
| `futo-tiny`  | 1,3 M     | 4  | 128  | 256   | —          | —          | un processeur, 45 s |
| `futo-mac`   | 39,3 M    | 8  | 512  | 1 024 | 1,5 G tokens| 0,43 EFLOP | un Mac, ~2,5 j (voir [docs/MAC.md](docs/MAC.md)) |
| `futo-small` | 100,7 M   | 12 | 768  | 1 024 | 10 G tokens| 7,1 EFLOP  | 1 GPU, ~5 h sur H100, 20-40 € |
| `futo-base`  | 299,4 M   | 24 | 1 024| 2 048 | 30 G tokens| 72 EFLOP   | 1 GPU, ~50 h sur H100, 150-250 € |
| `futo-large` | 1 180,8 M | 24 | 2 048| 2 048 | 100 G tokens| 829 EFLOP | 8 GPU, ~3 j, 1 500-2 500 € |

Les tailles sont calculées, pas approximées : un test échoue si une
configuration s'écarte de ce qui est annoncé. Les durées supposent 40 % de MFU
et sont donc optimistes — comptez 1,5 à 2 fois plus pour les petits modèles, où
le surcoût des noyaux fait tomber le MFU à 20-30 %.

Pour voir le détail d'une configuration avant de louer quoi que ce soit —
la commande commence par estimer la durée **sur votre propre machine** :

```bash
futo info configs/futo-small.yaml
```

Et pour remplacer l'estimation par une **mesure**, sur votre machine, en une
trentaine de secondes et sans corpus :

```bash
futo bench configs/futo-small.yaml
```

Sur un Mac Apple Silicon, le GPU intégré est détecté et utilisé sans réglage.
Un Mac ne mènera pas le run complet de `futo-small` (plus d'un mois), mais il
est le bon outil pour tout ce qui vient avant : entraîner le tokenizer,
préparer les shards, et comparer des mélanges de corpus avec `futo-mac`.
Voir [docs/MAC.md](docs/MAC.md).

Le budget de tokens visé est d'environ 100 tokens par paramètre, soit cinq fois
le ratio de Chinchilla. C'est délibéré : Chinchilla minimise le coût de
l'**entraînement**, alors qu'on veut ici minimiser celui de l'**inférence**. Un
modèle plus petit et plus longuement entraîné coûte moins cher à faire tourner
ensuite, et c'est ce qui compte quand on veut le déployer.

---

## Ce qui est réellement français là-dedans

L'architecture d'un transformeur décodeur n'a rien de national. Deux choses
portent le français : le tokenizer, et les données. Le reste est de la plomberie.

### Le tokenizer

Les tokenizers courants sont calibrés sur l'anglais, où la contraction se trouve
à **droite** de l'apostrophe : `do|n't`, `it|'s`. En français, l'élision est à
**gauche** : `l'|homme`, `qu'|il`, `aujourd'|hui`. Appliquer une règle
anglophone au français découpe `l` puis `'homme`, ce qui sépare l'apostrophe de
son proclitique et disperse le vocabulaire sur des formes qui n'ont aucune
existence linguistique.

Le motif de découpe de Futo garde l'élision soudée, et fait de même pour les
clitiques accrochés par un trait d'union :

```
L'homme qu'il a vu aujourd'hui  →  L' | homme |  qu' | il |  a |  vu |  aujourd' | hui
Y a-t-il quelqu'un ?            →  Y |  a | -t-il |  quelqu' | un |  ?
Donne-le-moi, va-t'en           →  Donne | -le | -moi | , |  va | -t'en
```

Autres décisions, toutes vérifiées par des tests :

- **Les deux apostrophes sont conservées.** Le français s'écrit tantôt avec `'`
  (U+0027), tantôt avec `’` (U+2019). Les confondre simplifierait le
  vocabulaire mais empêcherait le modèle de restituer le texte d'origine. Le
  coût réel se limite à quelques dizaines de tokens sur 32 768.
- **BPE au niveau octet.** Le vocabulaire de départ contient les 256 octets :
  rien ne peut être hors vocabulaire, et l'aller-retour texte → tokens → texte
  est exact. Vérifié sur les accents, les ligatures `œ` et `æ`, les guillemets
  français, les émojis et du code mêlé de français.
- **Normalisation NFC.** Sans elle, `é` composé et `é` décomposé donnent deux
  suites de tokens différentes pour le même mot.
- **Les chiffres sont coupés par groupes de trois.** Sinon le modèle apprend par
  cœur les nombres fréquents et calcule d'autant plus mal.

### L'évaluation

Un modèle de 100 à 400 millions de paramètres répond au niveau du hasard aux
grands bancs d'essai de connaissances. Le mesurer là-dessus ne dit rien.

La grammaire, en revanche, s'acquiert tôt et se mesure proprement. Futo est
livré avec **72 paires minimales françaises** — deux phrases quasi identiques
dont une seule est correcte — réparties sur 20 phénomènes :

```
correct   : Les lettres qu'elle a écrites sont restées sans réponse.
incorrect : Les lettres qu'elle a écrit sont restées sans réponse.

correct   : Nous partirons avant qu'il fasse nuit.
incorrect : Nous partirons avant qu'il fait nuit.

correct   : Le hérisson traversait la route sans se presser.
incorrect : L'hérisson traversait la route sans se presser.
```

Les cas **symétriques** sont inclus exprès. Sans les paires « COD placé après »
(où le participe ne s'accorde pas) et « indicatif attendu » (où le subjonctif
serait fautif), un modèle qui accorderait tout systématiquement obtiendrait un
score parfait sans rien avoir compris.

À côté, l'évaluation rapporte les **bits par octet** plutôt que la seule
perplexité. La perplexité dépend du tokenizer : un modèle qui découpe le
français en gros tokens a mécaniquement une perplexité par token plus élevée
sans être moins bon. Les bits par octet ramènent la mesure au texte brut, ce qui
la rend comparable entre n'importe quels modèles.

---

## Le modèle

Un décodeur pré-normalisé, dans ce qui s'est stabilisé depuis Llama :

| | |
|---|---|
| Normalisation | RMSNorm, en pré-norme, statistique calculée en float32 |
| Positions | RoPE, tables cos/sin en float32, θ = 10 000 |
| Attention | causale, requêtes groupées (GQA), via `scaled_dot_product_attention` |
| Réseau avant | SwiGLU, dimension cachée ≈ 8/3 · d_model arrondie à un multiple de 64 |
| Biais | aucun |
| Poids | embedding d'entrée et tête de sortie liés |

Écarté volontairement : les mélanges d'experts (complexité sans rapport avec la
taille visée), ALiBi (RoPE fait mieux), l'attention linéaire (gain nul à ces
longueurs de contexte).

Le fichier [`futo/model.py`](futo/model.py) documente six pièges classiques
d'implémentation. Le plus vicieux :

> `is_causal=True` n'est correct que si `q_len == kv_len`. Le masque de
> `scaled_dot_product_attention` est aligné en haut à gauche : avec un cache KV
> et une seule requête, il masquerait tout sauf le premier token. Le modèle
> s'entraîne alors parfaitement — l'entraînement, lui, a bien `q_len == kv_len` —
> et ne génère que du bruit. Le test `test_cache_kv_equivaut_a_la_passe_complete`
> existe pour ça.

---

## Reprendre après un plantage

Un pré-entraînement dure des jours et coûte des centaines d'euros. Un plantage à
80 % ne doit rien coûter.

```bash
futo train configs/futo-small.yaml --reprendre auto
```

La reprise est **exacte** : les poids finaux sont identiques au bit près à ceux
qu'on aurait obtenus sans interruption. C'est vérifié par un test.

Cela tient à un choix de conception : le chargeur de données ne garde **aucun
état**. Le lot du pas *n* découle entièrement de `(graine, n)`. Il n'y a donc
rien à sérialiser, et rien qui puisse se désynchroniser entre le checkpoint et
le flux de données — la panne classique des boucles d'entraînement. Le
checkpoint enregistre le reste : poids, état de l'optimiseur, numéro de pas, et
l'état des trois générateurs aléatoires (PyTorch, NumPy, Python).

Les sauvegardes sont atomiques (écriture puis renommage) : un plantage *pendant*
une sauvegarde ne laisse jamais un checkpoint tronqué — et c'est justement le
moment où l'on plante le plus, le disque étant sollicité.

---

## Structure du dépôt

```
futo/
  config.py      configurations typées, YAML avec héritage, surcharges --set
  model.py       le transformeur : RMSNorm, RoPE, GQA, SwiGLU, cache KV
  tokenizer.py   BPE au niveau octet, découpe adaptée au français
  data.py        format de shards, préparation, chargeur déterministe
  train.py       boucle d'entraînement, plannings, checkpoints, MFU, journal
  eval.py        perplexité, bits par octet, sondes grammaticales
  cli.py         la commande « futo », dont « bench » (débit réel mesuré)
configs/         les quatre tailles, plus les réglages communs
data/echantillon/  142 Ko de français original, pour les tests hors ligne
data/sondes/     les 72 paires minimales françaises
tests/           190 tests, 17 s sur processeur, sans réseau
docs/            architecture, données, carte du modèle, Mac
```

---

## Les tests

```bash
pytest
```

190 tests, 17 secondes sur quatre cœurs, aucun accès réseau. Les plus importants
ne vérifient pas des formes de tenseurs mais des propriétés qu'un modèle peut
violer en silence :

| Test | Ce qu'il empêche |
|---|---|
| `test_causalite_aucune_fuite_du_futur` | que le modèle triche en lisant le futur — la perte s'effondre, et le modèle est inutilisable |
| `test_cache_kv_equivaut_a_la_passe_complete` | qu'il s'entraîne bien mais génère du bruit |
| `test_rope_ne_depend_que_de_la_distance` | que les positions rotatives soient mal appariées |
| `test_aucun_token_perdu_ni_duplique` | qu'on entraîne sur autre chose que ce qu'on croit |
| `test_reprise_donne_exactement_la_meme_suite` | qu'un plantage coûte plusieurs jours de GPU |
| `test_surapprentissage_dun_lot` | qu'on entraîne des heures sans rien apprendre |
| `test_comptage_parametres_analytique` | qu'on loue le mauvais GPU |
| `test_aller_retour_exact` | que le tokenizer perde des accents ou des ligatures |
| `test_commentaires_shell_sans_apostrophe` | qu'un bloc du README bloque le terminal au copier-coller |

Deux d'entre eux ont attrapé de vrais défauts pendant l'écriture du dépôt.

---

## Feuille de route

Voir [ROADMAP.md](ROADMAP.md). En résumé, l'ordre des chantiers :

1. ✅ Le moteur, testé de bout en bout — c'est ce dépôt.
2. ⬜ **Les données.** Rassembler, filtrer et dédupliquer 10 milliards de tokens
   de français. C'est le vrai travail, et le seul qui décidera de la réussite.
3. ⬜ Entraîner `futo-small` et publier les chiffres, bons ou mauvais.
4. ⬜ Passer à `futo-base` si les mesures le justifient.
5. ⬜ `futo-large`, seulement si les trois étapes précédentes ont tenu leurs
   promesses.

---

## Licence

Le code est sous [Apache 2.0](LICENSE) : permissive, avec une clause de brevets
explicite, ce qui est plus sûr que MIT pour un projet d'apprentissage
automatique publié par une société.

Le corpus d'exemple de `data/echantillon/` a été **écrit pour ce dépôt** et
suit la même licence. Aucun texte publié n'y a été repris : c'est un choix
délibéré, il n'y a donc aucune question de droits à instruire.

La licence des **poids** d'un futur modèle dépendra des corpus employés pour
l'entraîner — Wikipédia en CC BY-SA n'impose pas les mêmes obligations qu'une
extraction de Common Crawl. À trancher avant l'entraînement, pas après :
voir [docs/DONNEES.md](docs/DONNEES.md).
