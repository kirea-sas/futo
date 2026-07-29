# Corpus d'exemple

Environ 142 Ko de français, répartis en trois fichiers :

| Fichier | Taille | Contenu |
|---|---|---|
| `registres-courants.txt` | 54,9 Ko | administratif, journalistique, correspondance professionnelle, petites annonces, commercial |
| `narratif-et-varia.txt` | 45,5 Ko | récits, descriptions, vulgarisation historique, expressions idiomatiques et tournures régionales, textes d'opinion |
| `technique-et-oral.txt` | 41,7 Ko | documentation informatique (avec du code), scientifique et médical, dialogues transcrits, recettes, notices |

## À quoi ça sert

À faire tourner les tests et le premier essai **hors ligne**, sans rien
télécharger : entraînement d'un tokenizer, préparation des shards, entraînement
d'un modèle jouet, génération.

## À quoi ça ne sert pas

À entraîner un modèle utile. Cent quarante-deux kilooctets représentent environ
39 000 tokens, quand `futo-small` en demande dix milliards — soit deux cent
cinquante mille fois plus. Un modèle entraîné là-dessus mémorise le corpus en
quelques centaines de pas et ne généralise rien. C'est visible dans le premier
essai du README : la perte d'entraînement descend pendant que celle de
validation remonte.

Pour un vrai corpus, voir [../../docs/DONNEES.md](../../docs/DONNEES.md) ou
lancer `futo data telecharger`.

## Provenance et licence

**Ces textes ont été écrits pour ce dépôt.** Aucun texte publié n'y a été repris :
ni citation d'auteur, ni extrait de presse, ni documentation recopiée. Les
personnes, entreprises, communes, logiciels et médicaments qui y figurent sont
inventés ; les seules villes réelles citées le sont dans des passages de
vulgarisation historique.

Ils suivent donc la licence du dépôt, Apache 2.0, et n'ouvrent aucune question
de droits.

## Ce qu'ils contiennent volontairement

Le corpus a été écrit pour éprouver la découpe du français. On y trouve, répartis
naturellement dans des textes suivis :

- les élisions (`l'homme`, `qu'il`, `aujourd'hui`, `jusqu'à`, `presqu'île`,
  `lorsqu'on`), y compris les formes orales (`j'sais pas`, `t'as vu`, `y'a`) ;
- **les deux apostrophes**, droite `'` et typographique `’`, volontairement
  mélangées ;
- les clitiques accrochés par un trait d'union (`dit-il`, `vas-y`, `y a-t-il`,
  `va-t'en`, `donne-le-moi`, `celui-ci`) ;
- les majuscules accentuées (`À É È Ê Ç Ô Û Ï Æ Œ`) et les ligatures (`cœur`,
  `œuvre`, `sœur`, `bœuf`, `vœux`, `ex æquo`, `nævus`) ;
- les guillemets français `« »` avec espaces insécables (U+00A0), et les
  espaces insécables avant `; : ! ?` ;
- les nombres à la française (`1 234,56 €`, `12 %`, `3,14`, `180 °C`, `2 h 30`,
  `1er`, `2e`, `XIXᵉ siècle`), les dates, les téléphones, les codes postaux ;
- les sigles et abréviations (`SNCF`, `INSEE`, `URSSAF`, `M.`, `Mme`, `Dr`,
  `n°`, `c'est-à-dire`, `etc.`, `cf.`) ;
- le subjonctif présent et imparfait, le passé simple, les participes passés
  accordés, les pronoms clitiques enchaînés (`je le lui ai donné`) ;
- du code (shell, Python, JavaScript, JSON, YAML) mêlé au français, parce que
  c'est un cas que le tokenizer doit savoir segmenter.
