# Les données

C'est le chantier qui décidera de la réussite de Futo. Le calcul est un problème
résolu : on loue des GPU, on connaît le prix. Rassembler dix milliards de tokens
de français propre, non — et aucun code de ce dépôt ne le fera à votre place.

> **Avertissement sur ce document.** Les noms de corpus et les ordres de
> grandeur ci-dessous sont donnés **de mémoire**. Les identifiants exacts, les
> volumes et surtout les licences changent, et une erreur sur une licence se
> paie cher. Chaque ligne marquée **[À VÉRIFIER]** doit être contrôlée sur la
> fiche officielle du jeu de données avant tout téléchargement. Considérez ce
> document comme une carte, pas comme un GPS.

---

## Combien en faut-il

| Modèle | Paramètres | Budget visé | Volume brut approximatif |
|---|---|---|---|
| `futo-small` | 100,7 M | 10 G tokens | ~36 Go de texte UTF-8 |
| `futo-base` | 299,4 M | 30 G tokens | ~110 Go |
| `futo-large` | 1 180,8 M | 100 G tokens | ~360 Go |

La conversion utilise le rapport mesuré sur le corpus d'exemple avec le
tokenizer de Futo : **3,6 octets par token** en français. Ce chiffre est réel,
pas estimé — mais il est mesuré sur 142 Ko de texte soigné. Sur du web brut,
plus bruité, attendez-vous à un rapport un peu moins bon.

À titre de repère : la Wikipédia francophone représente quelques gigaoctets de
texte. Elle ne suffit donc pas, à elle seule, même pour `futo-small`.

---

## Les sources

### Web filtré — le gros du volume

**FineWeb-2** **[À VÉRIFIER]** — extraction de Common Crawl filtrée et
dédupliquée, déclinée par langue, dont le français. C'est aujourd'hui le
meilleur rapport volume/qualité pour du texte web, et le point de départ le plus
raisonnable. Vérifiez l'identifiant exact, la taille de la portion française et
la licence.

**CulturaX**, **OSCAR**, **HPLT** **[À VÉRIFIER]** — autres extractions
multilingues de Common Crawl, plus anciennes ou moins filtrées. Utiles pour
compléter le volume, mais demandent un filtrage supplémentaire. Leurs qualités
respectives varient selon les millésimes.

### Textes soignés — la qualité

**Wikipédia en français** — la source propre par excellence : encyclopédique,
relue, bien structurée, et **le point de départ recommandé**. C'est aussi la
seule source pour laquelle Futo fournit l'outillage complet :

```bash
futo data wikipedia frwiki-latest-pages-articles.xml.bz2
```

Si le dump n'est pas déjà sur le disque, la commande le télécharge elle-même :

```bash
futo data wikipedia --telecharger
```

Environ 7 Gio, plus d'une heure sur une liaison domestique. Une coupure n'est
pas grave : relancer la commande reprend à l'octet près, grâce à l'en-tête HTTP
`Range`, et un fichier déjà complet n'est jamais retéléchargé.

Le dump se récupère sinon à la main sur `dumps.wikimedia.org`, dans le fichier
`frwiki-latest-pages-articles.xml.bz2`. La commande le
lit en flux — jamais en mémoire —, nettoie le wikitexte (modèles imbriqués,
tableaux, liens, références, catégories, sections de fin), écrit le JSONL
attendu par `futo data preparer`, et consigne la source dans `data/SOURCES.md`.
Essayez d'abord sur un échantillon : `--articles-max 1000`.

Le nettoyage préserve ce qui fait le français : élisions, apostrophes
typographiques, ligatures, guillemets et espaces insécables. Il est testé sur
des fragments écrits à la main, **jamais sur un vrai dump** — les surprises
viendront de là.

**La licence est tranchée** : CC BY-SA impose l'attribution et le partage à
l'identique, et l'effet sur des poids entraînés n'est pas clair juridiquement.
Décision prise le 29/07/2026, avant tout entraînement : **les poids de Futo
seront publiés sous CC BY-SA 4.0**, comme les données. L'usage commercial reste
permis ; l'attribution et le partage à l'identique s'imposent à qui redistribue.
Le code reste sous Apache 2.0. Voir [CARTE-DU-MODELE.md](CARTE-DU-MODELE.md).

Conséquence pour les autres sources : si l'on ajoute un jour un corpus dont la
licence est incompatible avec CC BY-SA, il faudra le vérifier avant de le
mélanger, pas après.

**Wikisource, Gallica, Projet Gutenberg** **[À VÉRIFIER]** — littérature du
domaine public. Excellent français, mais daté : un modèle nourri
majoritairement de textes du XIXᵉ siècle écrira comme le XIXᵉ siècle. À doser.

**Corpus institutionnels français** **[À VÉRIFIER]** — Légifrance et le Journal
officiel, les données de data.gouv.fr, les thèses, les débats parlementaires
européens. Français administratif et technique de bonne tenue, licences
généralement ouvertes, mais registre très particulier : à doser également.

**Les corpus de l'initiative OpenLLM-France** **[À VÉRIFIER]** — plusieurs
projets francophones ont publié des corpus d'entraînement pensés pour le
français, avec un travail de sélection et de documentation déjà fait. C'est la
première piste à explorer avant de tout refaire soi-même.

### Anglais et code

Un modèle purement français est un choix défendable, mais coûteux en pratique :
la documentation technique, le code et une grande partie du raisonnement
disponible sont en anglais. Un peu d'anglais et de code améliore les capacités
générales, y compris en français.

---

## Un mélange de départ

À défaut de pouvoir mesurer avant d'avoir essayé, voici une répartition de
départ défendable pour un modèle français généraliste :

| Part | Source | Passes |
|---|---|---|
| 55 % | Web français filtré | 1 |
| 15 % | Wikipédia française | 2 |
| 10 % | Textes institutionnels et techniques français | 1 |
| 5 % | Littérature du domaine public | 1 |
| 10 % | Anglais de bonne qualité | 1 |
| 5 % | Code | 1 |

**Ce tableau est une hypothèse de travail, pas un résultat.** La seule façon de
le valider est d'entraîner plusieurs `futo-small` avec des mélanges différents
et de comparer les bits par octet et les sondes grammaticales. C'est
précisément ce que `futo-small` est fait pour permettre : à 20-40 € le run, on
peut se payer quatre ou cinq mélanges avant de lancer `futo-base`.

---

## Le filtrage

**Le piège principal : les filtres publiés sont calibrés sur l'anglais.** Les
seuils de Gopher et de C4 sont repris partout sans être ré-étalonnés, et
plusieurs se comportent mal sur le français :

- **Longueur moyenne des mots.** Le français a des mots plus longs que
  l'anglais. Un seuil anglophone rejette du français parfaitement correct.
- **Filtres par mots vides.** Il faut la liste française (`le`, `la`, `les`,
  `de`, `des`, `et`, `à`, `un`, `une`, `est`, `que`, `qui`, `dans`, `pour`), pas
  celle de l'anglais. C'est une erreur qui vide silencieusement un corpus.
- **Ratio de ponctuation.** L'apostrophe est bien plus fréquente en français
  (élisions), les guillemets sont différents (`« »`), et les espaces
  insécables comptent comme des caractères. Un filtre naïf pénalise le bon
  français typographié.
- **Détection de langue.** Le français, l'occitan, le catalan et le wallon se
  confondent sur des textes courts. Filtrez sur des documents entiers, pas sur
  des phrases isolées.
- **Filtre de « qualité » par perplexité.** Il faut un modèle de référence
  français ; un modèle anglophone rejettera tout.

Étapes recommandées, dans l'ordre :

1. **Normalisation Unicode NFC** — la même règle qu'à l'entraînement du
   tokenizer, sans quoi les accents se dédoublent.
2. **Détection de langue** sur le document entier.
3. **Filtres heuristiques**, ré-étalonnés sur du français : longueur minimale,
   proportion de lettres, lignes dupliquées, densité de ponctuation.
4. **Déduplication exacte** par empreinte du document — c'est peu coûteux et
   cela retire déjà l'essentiel.
5. **Déduplication approchée** (MinHash) si le volume le justifie. Les corpus
   web contiennent énormément de quasi-doublons, et un modèle qui voit dix fois
   le même texte le mémorise au lieu d'apprendre.
6. **Décontamination** vis-à-vis des jeux d'évaluation. Retirez au minimum les
   phrases du fichier `data/sondes/paires-minimales-fr.jsonl` : les laisser dans
   l'entraînement rendrait la mesure grammaticale sans valeur.

---

## Le format attendu

Un fichier JSONL — éventuellement compressé en `.gz` — un document par ligne,
avec un champ `text` :

```json
{"text": "Le premier document, en français.\n\nAvec ses paragraphes."}
{"text": "Le deuxième document."}
```

Puis :

```bash
futo tokenizer entrainer --corpus 'data/brut/*.jsonl' --vocab 32768 \
                         --octets-max 2000000000
futo data preparer --corpus 'data/brut/*.jsonl' --separateur jsonl
```

Inutile d'entraîner le BPE sur plus de 2 Go : au-delà, on paie du temps sans
gagner en qualité de vocabulaire. D'où `--octets-max`.

Les shards produits sont décrits en tête de [`futo/data.py`](../futo/data.py) :
en-tête de 1 024 octets auto-descriptif, puis les tokens en `uint16`. Le champ
« taille du vocabulaire » de l'en-tête est un garde-fou — réutiliser par erreur
des shards préparés avec un autre tokenizer produirait un modèle silencieusement
incohérent, et le chargeur refuse de le faire.

---

## Droit d'auteur, données personnelles, AI Act

Trois obligations distinctes, à traiter avant l'entraînement.

**Le droit d'auteur.** En France, l'exception de fouille de textes et de données
(article L122-5-3 du code de la propriété intellectuelle, transposant la
directive européenne 2019/790) autorise les copies nécessaires à la fouille pour
la recherche, et pour les autres usages tant que les ayants droit ne s'y sont
pas opposés. Cette opposition passe en pratique par les fichiers `robots.txt` et
les métadonnées des sites. Les grands corpus publics affirment généralement en
tenir compte — **[À VÉRIFIER]** pour chacun, c'est exactement le genre
d'affirmation qu'il faut contrôler plutôt que croire.

**Les données personnelles.** Un corpus web contient des noms, des adresses, des
numéros de téléphone. Le RGPD s'applique. À prévoir : un repérage et une
neutralisation des données personnelles les plus évidentes (adresses
électroniques, numéros de téléphone, numéros de sécurité sociale), et une
procédure permettant de répondre à une demande d'effacement — sujet
inconfortable pour un modèle déjà entraîné, mais qui ne disparaît pas si on
l'ignore.

**Le règlement européen sur l'IA.** Il impose aux fournisseurs de modèles à
usage général de publier un résumé suffisamment détaillé des contenus
d'entraînement, selon un modèle fourni par le Bureau européen de l'IA. Concevez
la documentation des sources comme un livrable dès le premier téléchargement :
reconstituer cette liste après coup est pénible, et souvent impossible.

**En pratique :** tenez un fichier `data/SOURCES.md` où chaque téléchargement
est consigné le jour même — nom exact, identifiant, version, date, licence,
volume obtenu. C'est cinq minutes sur le moment, et une journée perdue six mois
plus tard.
