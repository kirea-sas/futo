# Reproduire Futo de bout en bout

Ce document permet de refaire **tout** le travail, depuis un dépôt vierge
jusqu'à un modèle entraîné et mesuré, sans rien connaître du projet et sans
poser de question à personne. Chaque commande est donnée telle qu'elle a été
lancée, chaque durée est celle qui a été relevée, chaque piège rencontré est
signalé à l'endroit où il se présente.

Publié par **Kirea SAS** — [kirea.fr](https://kirea.fr) — sous licence Apache 2.0
pour le code, CC BY-SA 4.0 pour les poids.

---

## Sommaire

1. [Ce que vous obtiendrez](#1-ce-que-vous-obtiendrez)
2. [Matériel et durées relevées](#2-materiel-et-durees-relevees)
3. [Installation](#3-installation)
4. [Le corpus](#4-le-corpus)
5. [Le tokenizer](#5-le-tokenizer)
6. [Les shards](#6-les-shards)
7. [L'entraînement](#7-lentrainement)
8. [L'évaluation](#8-levaluation)
9. [Reprendre après une coupure](#9-reprendre-apres-une-coupure)
10. [Pièges rencontrés](#10-pieges-rencontres)
11. [Vérifier que vous avez le même résultat](#11-verifier-que-vous-avez-le-meme-resultat)

---

## 1. Ce que vous obtiendrez

Un modèle de langue français de 39,3 millions de paramètres, entraîné sur
1,49 milliard de tokens de Wikipédia francophone, avec :

- un tokenizer BPE au niveau octet, découpe adaptée au français
- les poids entraînés et les points de reprise intermédiaires
- des mesures : perplexité, bits par octet, sondes grammaticales, suggestions

Ce modèle n'est **pas** destiné à être utile en production. Il sert à comparer :
deux mélanges de corpus, deux tailles de vocabulaire, deux réglages. À cette
échelle, un aller-retour tient dans une nuit, et les conclusions se transposent
le plus souvent à un modèle dix fois plus gros. C'est le bon endroit pour se
tromper, avant de payer un GPU.

Prévoyez :

| Ressource | Quantité |
|---|---|
| Espace disque | 20 Go (dump 7,3 Go, JSONL 7,1 Go, shards 3,3 Go, points de reprise) |
| Mémoire vive | 16 Go suffisent |
| Temps machine | environ 60 heures au total |
| Temps humain | une heure, réparties sur les étapes |

---

## 2. Matériel et durées relevées

Toutes les durées ci-dessous ont été **mesurées**, sur un MacBook Pro Apple
M2 Max. Elles ne sont pas extrapolées.

| Étape | Durée relevée | Machine |
|---|---|---|
| Téléchargement du dump (7,3 Go) | variable, selon la ligne | — |
| Conversion du dump en JSONL | environ 2 h | M2 Max |
| Entraînement du tokenizer | 10 min | M2 Max |
| Préparation des shards | environ 1 h | M2 Max |
| Entraînement complet | environ 50 h | M2 Max, MPS, bfloat16 |

Débit relevé pendant l'entraînement : 8 700 tokens/s, MFU 18,4 %, 15,0 s par pas.

Sur une autre machine, mesurez la vôtre plutôt que d'extrapoler celle-ci — voir
[l'étape 7](#7-lentrainement).

---

## 3. Installation

Python 3.10, 3.11 ou 3.12. Rien d'autre n'est requis : aucune dépendance
système, aucun compilateur.

```bash
git clone https://github.com/kirea-sas/futo.git
cd futo
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Le `venv` n'est pas une coquetterie. Les distributions récentes appliquent la
PEP 668 et refusent tout `pip install` hors environnement virtuel. Sans lui,
`pip` puis `futo` seront introuvables.

Vérifiez :

```bash
futo info configs/futo-mac.yaml
```

Cela affiche la taille du modèle, le budget en tokens, le matériel détecté et
une estimation de durée. Aucun calcul lourd n'est lancé.

> À chaque nouvelle session de terminal, il faut refaire `source .venv/bin/activate`.
> Sans cela, `futo` reste introuvable. C'est la cause numéro un des
> « command not found ».

Lancez la suite de tests, elle tourne hors ligne en une vingtaine de secondes :

```bash
pip install -e ".[dev]"
pytest -q
```

---

## 4. Le corpus

### 4.1 Télécharger le dump

```bash
futo data wikipedia --telecharger --vers data/brut/frwiki-articles.xml.bz2
```

7,3 Go. Le téléchargement reprend à l'octet près après une coupure : relancez la
même commande, rien n'est retéléchargé.

### 4.2 Contrôler AVANT de tout convertir

Convertir 2,5 millions d'articles prend deux heures. Vérifiez d'abord la
propreté sur un échantillon :

```bash
futo data wikipedia data/brut/frwiki-articles.xml.bz2 \
    --sortie data/brut/essai-1000.jsonl --articles-max 1000
futo data controler data/brut/essai-1000.jsonl
```

Le verdict attendu est **propre**, avec moins de 0,1 % de documents portant une
trace de balisage. Si le taux dépasse quelques pour cent, ne lancez pas la
conversion complète : le nettoyeur a rencontré une construction qu'il ne sait
pas traiter, et il vaut mieux le corriger avant.

Lisez aussi les extraits que la commande affiche. Les compteurs ne voient pas
tout : lors de la mise au point, deux défauts n'ont été trouvés qu'à la lecture
— des titres de sections consécutives qui ressortaient collés, et des modèles de
date supprimés qui laissaient « né le à Moulins », du français agrammatical.

### 4.3 Convertir tout le dump

```bash
futo data wikipedia data/brut/frwiki-articles.xml.bz2 \
    --sortie data/brut/wikipedia-fr.jsonl
```

Environ 2 h. La lecture se fait en flux, jamais en mémoire : le processus tient
dans quelques centaines de mégaoctets quelle que soit la taille du dump.

Résultat relevé : **2 506 250 documents, 7,14 Go de JSONL**.

La commande écrit aussi `data/SOURCES.md` : provenance, date, licence, volume.
Ce n'est pas de la bureaucratie — le règlement européen sur l'IA impose de
publier un résumé des contenus d'entraînement, et cette information est
impossible à reconstituer après coup.

Contrôlez le résultat complet :

```bash
futo data controler data/brut/wikipedia-fr.jsonl --documents-max 10000
```

---

## 5. Le tokenizer

```bash
futo tokenizer entrainer \
    --corpus data/brut/wikipedia-fr.jsonl \
    --vocab 32768 \
    --octets-max 2000000000
```

10 minutes. Deux milliards d'octets suffisent largement pour un BPE ; lire les
7,1 Go entiers ne changerait pratiquement rien au vocabulaire obtenu et
coûterait beaucoup plus de temps.

Résultats relevés :

```
vocabulaire obtenu : 32768
fertilité mesurée  : 1,561 token/mot · 4,45 octet/token
                     73,4 % des mots en un seul token
```

Vérifiez la découpe sur une phrase piégeuse :

```bash
futo tokenizer info data/tokenizer/futo-tokenizer.json \
    --texte "L'homme qu'elle avait rencontré aujourd'hui n'était pas celui-ci."
```

Attendu : les élisions restent soudées au mot qui les porte — `L'`, ` qu'`,
` aujourd'`, ` n'` — et le clitique `-ci` reste attaché. C'est tout l'objet de
la découpe française : un tokenizer généraliste sépare l'apostrophe du mot et
gaspille un token à chaque élision, ce qui est très fréquent en français.

---

## 6. Les shards

```bash
futo data preparer --corpus data/brut/wikipedia-fr.jsonl
```

Environ 1 h. Écrit des shards binaires auto-descriptifs dans `data/prepare` :
un en-tête de 1024 octets puis les tokens bruts en uint16, lus ensuite par
projection mémoire sans jamais tout charger.

Résultats relevés :

```
2 506 250 documents · 1 650 819 778 tokens
17 shards d'entraînement, 10 002 255 tokens de validation
4,44 octet/token
```

Les 4,44 octets par token confirment la mesure faite à l'étape précédente sur un
autre échantillon : c'est un bon signe de cohérence.

Inspectez ce qui a été écrit :

```bash
futo data info data/prepare
```

---

## 7. L'entraînement

### 7.1 Mesurer votre machine d'abord

```bash
futo bench configs/futo-mac.yaml
```

Deux minutes, aucun corpus nécessaire. Cela donne le débit réel en tokens par
seconde et la durée du run complet **sur votre matériel**. Les durées affichées
par `futo info` pour les autres machines sont, elles, calculées et non mesurées.

### 7.2 Lancer

```bash
futo train configs/futo-mac.yaml
```

Environ 50 h sur M2 Max. Le journal s'écrit dans `sorties/mac/`, au format
JSONL et CSV, et un point de reprise est enregistré tous les 500 pas.

Ce qu'il faut lire, ligne par ligne :

```
pas    500 · perte 4.1063 · lr 8.00e-04 · |g| 0.46 · 8 754 tok/s · MFU 18.4 %
```

- **perte** : élevée à la puissance e, elle donne le nombre de mots entre
  lesquels le modèle hésite encore. Au premier pas, elle doit valoir environ
  10,40, soit le logarithme naturel de 32 768 : un modèle qui ne sait rien
  répartit ses chances également sur tout le vocabulaire. La voir démarrer
  ailleurs signale un problème d'initialisation.
- **lr** : monte de zéro au maximum sur 300 pas, reste au plateau, puis descend
  sur les 1 500 derniers. Un modèle neuf corrigé trop fort ne revient jamais.
- **|g|** : la norme du gradient. Elle décroît et se stabilise. Si elle explose,
  quelque chose casse.
- **MFU** : la part de la puissance crête réellement employée.

Repères relevés lors du run de référence :

| Pas | Perte | Validation |
|---|---|---|
| 0 | 10,49 | — |
| 500 | 4,11 | 4,35 · perplexité 77,7 |
| 3 370 | 3,07 | — |
| 5 940 | 3,02 | — |

L'écart faible entre perte d'entraînement et perte de validation indique que le
modèle généralise au lieu d'apprendre par cœur. C'est attendu sur une seule
époque, mais il vaut mieux le constater que le supposer.

---

## 8. L'évaluation

```bash
futo eval sorties/mac/dernier.pt
```

Affiche quatre mesures :

- **la perplexité** sur les tokens de validation, jamais vus
- **les bits par octet**, la seule mesure indépendante du tokenizer, donc la
  seule comparable à d'autres modèles
- **les sondes grammaticales** : 72 paires minimales françaises, deux phrases
  quasi identiques dont une seule est correcte, le modèle doit désigner la bonne
- **les suggestions** : le mot attendu figure-t-il dans les k plus probables

Deux chiffres de suggestions sont donnés. Sur tous les tokens, la mesure est
flatteuse : elle compte aussi les fins de mots déjà entamés, et compléter
« aujourd' » par « hui » n'a rien d'un exploit. Sur les seuls débuts de mot,
elle mesure ce que fait vraiment un clavier à suggestions. C'est le second
chiffre qu'il faut publier.

Faites-lui écrire du texte :

```bash
futo generer sorties/mac/dernier.pt --amorce "La capitale de la France est"
```

Attendez-vous à du français correct qui n'affirme rien de fiable. À cette
échelle, le modèle apprend à parler, pas à savoir.

---

## 9. Reprendre après une coupure

Coupure de courant, batterie vide, interruption au clavier : rien n'est perdu
au-delà du dernier point de reprise.

```bash
futo train configs/futo-mac.yaml --reprendre
```

Vérifiez la première ligne affichée : elle doit indiquer le pas atteint, pas le
pas zéro. Si elle affiche zéro, arrêtez immédiatement — vous écraseriez le
travail fait.

Le tirage des lots est reconstruit à partir de la graine et du numéro de pas :
aucun état n'est sérialisé, et le modèle revoit exactement le texte prévu, sans
doublon ni saut. C'est ce qui rend la reprise exacte plutôt qu'approximative.

Si les coupures se répètent, écrivez plus souvent :

```bash
futo train configs/futo-mac.yaml --reprendre --set train.save_every=100
```

---

## 10. Pièges rencontrés

Tous ceux qui suivent ont réellement été rencontrés pendant la mise au point.

**`pip: command not found`, puis `futo: command not found`.** L'environnement
virtuel n'est pas activé. Refaites `source .venv/bin/activate` à chaque nouvelle
session de terminal. C'est la cause la plus fréquente, et de loin.

**Le terminal reste bloqué sur `quote>` sous zsh.** Une apostrophe dans un
commentaire shell copié-collé ouvre une chaîne que rien ne referme. Faites
Ctrl-C. La documentation de ce dépôt est testée contre ce défaut, mais pas celle
du reste du monde.

**Le tokenizer apprend du JSON.** Si le corpus est un fichier JSONL et que le
lecteur donne les lignes brutes au trainer, le vocabulaire se remplit de
`{"text": "` et de `", "titre": "`. Ce sont des places prises sur 32 768, et du
balisage que le modèle recrachera. Le lecteur du dépôt détecte le JSONL par son
extension et extrait le champ texte ; si vous adaptez le code, vérifiez ce point
en premier. Contrôle rapide, aucune fusion du vocabulaire ne doit contenir
d'accolade ou de guillemet.

**« Entraînement du tokenizer sur 7 321 453 631 octets » avec `--octets-max`.**
Le chiffre annoncé est la taille du fichier sur le disque, pas ce qui sera lu.
Depuis, la commande affiche les deux séparément.

**Le débit s'effondre sur un portable.** Une chute de 8 700 à 2 900 tokens/s
sur Mac signale un bridage du GPU par le système : batterie faible ou surchauffe.
Branchez sur secteur. Ce n'est pas un défaut du programme.

**La reprise échoue avec `RNG state must be a torch.ByteTensor`.** Défaut
corrigé, mais si vous portez ce code ailleurs : l'état du générateur aléatoire
doit être ramené sur le processeur avant restauration, sinon `map_location`
le déplace sur l'accélérateur et la restauration refuse.

**Un test de déterminisme qui exige le bit près.** Sur MPS comme sur CUDA,
l'ordre des réductions varie d'une exécution à l'autre. Une tolérance de 1e-5
hors processeur est la seule position tenable.

---

## 11. Vérifier que vous avez le même résultat

Les chiffres ci-dessous ont été relevés sur le run de référence. Un écart de
quelques pour cent est normal — l'ordre des réductions varie d'un accélérateur à
l'autre. Un écart d'un ordre de grandeur signale un problème.

| Étape | Attendu |
|---|---|
| Articles convertis | 2 506 250 |
| JSONL produit | 7,14 Go |
| Documents portant une trace de balisage | moins de 0,1 % |
| Vocabulaire | 32 768 exactement |
| Fertilité | 1,56 token/mot · 4,45 octet/token |
| Mots en un seul token | 73,4 % |
| Tokens préparés | 1 650 819 778 |
| Tokens de validation | 10 002 255 |
| Octets par token sur les shards | 4,44 |
| Perte au pas 0 | 10,49 |
| Perte de validation au pas 500 | 4,35 |

Si le vocabulaire obtenu est inférieur à celui demandé, le corpus lu est trop
petit : vérifiez `--octets-max` et le chemin du corpus.

Si la perte au pas 0 s'écarte nettement de 10,49, le vocabulaire du modèle et
celui du tokenizer ne correspondent pas. Comparez `model.vocab_size` dans la
configuration au vocabulaire réellement obtenu.

---

## Pour aller plus loin

- [ARCHITECTURE.md](ARCHITECTURE.md) — ce que fait le modèle, et pourquoi ces choix
- [DONNEES.md](DONNEES.md) — les sources, leurs licences, et celles qui posent problème
- [MAC.md](MAC.md) — spécificités d'Apple Silicon
- [CARTE-DU-MODELE.md](CARTE-DU-MODELE.md) — ce qu'il faut publier avec des poids
- [../ROADMAP.md](../ROADMAP.md) — ce qui est fait, ce qui reste, et les chiffres mesurés
