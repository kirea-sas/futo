# Carte du modèle — modèle à remplir

> **Aucun modèle Futo n'a été entraîné à ce jour.** Ce document est le gabarit à
> remplir lors de la première publication de poids. Il est versionné dès
> maintenant pour une raison simple : les informations qu'il demande — sources
> exactes, volumes, dates, consommation — sont faciles à noter sur le moment et
> pénibles, souvent impossibles, à reconstituer six mois plus tard.
>
> Les champs entre chevrons `<…>` sont à compléter. Un champ qu'on ne sait pas
> remplir doit être écrit « inconnu », jamais deviné.

---

## Identité

| | |
|---|---|
| Nom | `<futo-small-v1>` |
| Version | `<1.0>` |
| Date de publication | `<AAAA-MM-JJ>` |
| Éditeur | Kirea SAS |
| Licence du code | Apache 2.0 |
| Licence des poids | `<à trancher selon les corpus employés — voir plus bas>` |
| Contact | contact@kirea.fr |

## Architecture

| | |
|---|---|
| Type | transformeur décodeur causal |
| Paramètres | `<…>` |
| Couches / `d_model` / têtes | `<…>` |
| Contexte | `<…>` tokens |
| Vocabulaire | `<…>` |
| Précision d'entraînement | `<bf16>` |

Le détail est dans [ARCHITECTURE.md](ARCHITECTURE.md). La configuration exacte
est enregistrée dans le checkpoint lui-même : `futo eval <checkpoint>` l'affiche.

## Données d'entraînement

Le règlement européen sur l'IA impose de publier un résumé suffisamment détaillé
des contenus d'entraînement. Cette section est ce résumé — elle n'est pas
facultative.

| Source | Identifiant / version | Part du mélange | Passes | Licence | Date de collecte |
|---|---|---|---|---|---|
| `<…>` | `<…>` | `<…> %` | `<…>` | `<…>` | `<…>` |

| | |
|---|---|
| Volume total | `<…>` tokens |
| Langues | `<français …  %, anglais … %, code … %>` |
| Filtrage appliqué | `<résumé, et renvoi au script exact et à son empreinte git>` |
| Déduplication | `<exacte / MinHash, paramètres>` |
| Décontamination | `<jeux d'évaluation retirés — au minimum data/sondes/paires-minimales-fr.jsonl>` |

## Entraînement

| | |
|---|---|
| Matériel | `<n × H100 80 Go>` |
| Durée | `<… heures>` |
| MFU mesuré | `<… %>` |
| Coût de location | `<… €>` |
| Tokens vus | `<…>` |
| Optimiseur | AdamW, β = (0,9 · 0,95), weight decay 0,1 hors normes |
| Planning | `<cosine / wsd>`, échauffement `<…>` pas |

## Résultats

Chiffres **mesurés**. Un résultat non mesuré ne figure pas dans ce tableau.

| Mesure | Valeur | Sur quoi |
|---|---|---|
| Perte de validation | `<…>` | jeu de validation tenu à l'écart |
| Perplexité | `<…>` | idem |
| **Bits par octet** | `<…>` | `<corpus français tenu à l'écart>` |
| Sondes grammaticales | `<…> %` (hasard : 50 %) | les 72 paires de `data/sondes/` |
| Fertilité du tokenizer | `<…>` token/mot | `<corpus de contrôle>` |

Le détail par phénomène grammatical est produit par `futo eval` : le coller ici
tel quel, y compris les phénomènes ratés.

### Comparaisons

`<Comparer aux modèles français ou multilingues de taille voisine, en bits par
octet sur le MÊME texte — c'est la seule comparaison honnête entre modèles
n'ayant pas le même tokenizer. Indiquer précisément quel texte, et vérifier
qu'il n'appartient à aucun corpus d'entraînement des modèles comparés.>`

## Limites

À remplir sans complaisance. Ce qui est attendu ici :

- **Taille.** Un modèle de quelques centaines de millions de paramètres ne
  raisonne pas, ne calcule pas de façon fiable, et invente des faits avec
  aplomb. Le dire explicitement.
- **Domaines mal couverts.** `<…>`
- **Français non standard.** Registres régionaux, oral, argot, langues et
  créoles de France : `<couverture réelle dans le corpus>`
- **Contexte.** Au-delà de `<…>` tokens, la qualité se dégrade : RoPE
  n'extrapole pas de lui-même.
- **Pas de réglage par instructions.** `<le cas échéant>` Un modèle de base
  complète du texte, il ne suit pas de consignes et n'a reçu aucun alignement.

## Biais

`<Un corpus web français porte les biais de ce qui s'écrit en français sur
Internet : sur-représentation de certaines régions, de certains milieux, de
certaines opinions ; stéréotypes de genre inscrits jusque dans la grammaire —
le masculin générique, les métiers accordés par défaut. Décrire ce qui a été
mesuré, et dire franchement ce qui ne l'a pas été.>`

## Usages déconseillés

- Toute décision affectant une personne sans supervision humaine : recrutement,
  crédit, santé, justice, attribution de prestations.
- La production de texte présenté comme factuel sans vérification.
- Tout usage relevant des pratiques interdites ou à haut risque du règlement
  européen sur l'IA.

## Empreinte environnementale

| | |
|---|---|
| Énergie consommée | `<… kWh>` |
| Intensité carbone du réseau | `<… g CO₂e/kWh, selon le pays du centre de données>` |
| Émissions estimées | `<… kg CO₂e>` |
| Méthode | `<puissance de la carte × durée × PUE, ou relevé du fournisseur>` |

Un entraînement en France bénéficie d'un réseau électrique peu carboné : le
préciser plutôt que de reprendre une moyenne mondiale, qui serait pessimiste
d'un facteur cinq à dix.

## Licence des poids

À trancher **avant** l'entraînement, pas après, car le choix dépend des corpus :

- corpus entièrement permissifs → Apache 2.0, cohérent avec le code ;
- présence significative de contenus en CC BY-SA (Wikipédia) → l'effet du
  partage à l'identique sur des poids entraînés est un point juridique non
  tranché. Deux options défendables : publier sous CC BY-SA par prudence, ou
  documenter précisément la position retenue.

Dans tous les cas, la liste des sources ci-dessus doit être publiée avec les
poids.

## Reproduire

```bash
git checkout <empreinte git>
futo tokenizer entrainer --corpus <…> --vocab 32768
futo data preparer --corpus <…> --separateur jsonl
futo train configs/<…>.yaml
```

Indiquer l'empreinte git exacte, les graines (`train.seed`, `data.seed`) et la
version de PyTorch. La reproduction au bit près n'est pas garantie d'un matériel
à l'autre ; elle l'est sur le même matériel avec `train.deterministe: true`.
