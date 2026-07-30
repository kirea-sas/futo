# ROADMAP — Futo

Suivi des chantiers. Un à la fois, coché quand c'est fait et **mesuré**.

## ✅ Fait — le moteur

- [x] Modèle décodeur : RMSNorm, RoPE, GQA, SwiGLU, poids liés, cache KV
- [x] Tokenizer BPE au niveau octet, découpe adaptée au français (élisions et
      clitiques soudés, deux apostrophes conservées, chiffres par groupes de trois)
- [x] Format de shards auto-descriptif, chargeur sans état, reprise exacte
- [x] Boucle d'entraînement : AdamW, trois plannings, bf16/fp16/fp32,
      accumulation de gradient, DDP, MFU, journal JSONL + CSV
- [x] Évaluation : perplexité, bits par octet, 72 paires minimales françaises
- [x] Ligne de commande complète (`info`, `tokenizer`, `data`, `train`, `eval`, `generer`)
- [x] Cinq configurations chiffrées et vérifiées par test (1,3 M / 39,3 M /
      100,7 M / 299,4 M / 1 180,8 M)
- [x] Prise en charge du Mac (MPS) : détection de la puce, précision mixte
      essayée puis repli annoncé, configuration `futo-mac` — voir docs/MAC.md
- [x] Corpus d'exemple original de 142 Ko, licence propre, pour les tests hors ligne
- [x] 279 tests, 17 s sur processeur, sans réseau — CI GitHub Actions,
      et le démarrage du README rejoué depuis un clone neuf dans un
      environnement virtuel vierge, jusqu'à la génération de texte,
      dont des tests de la DOCUMENTATION : les blocs de commandes doivent
      être copiables-collables (une apostrophe dans un commentaire shell
      bloque zsh sur « quote> »), les liens internes doivent pointer
      quelque part, et les sous-commandes citées doivent exister

## 🔥 Prochain chantier — les données

C'est **le** sujet. Le reste est de la plomberie déjà écrite.

- [x] **Contrôle qualité d'un corpus** : `futo data controler <corpus>`
      compte les traces de balisage (accolades, crochets, balises, entités)
      avec un exemple de chacune, mesure la typographie française, repère
      les doublons exacts, et rend un verdict. À lancer sur un échantillon
      AVANT d'engager une conversion complète
- [x] **Téléchargement du dump intégré** : `futo data wikipedia --telecharger`
      récupère les ~7 Gio, avec reprise à l'octet près après une coupure et
      aucun retéléchargement d'un fichier complet
- [x] **Ingestion de Wikipédia** : `futo data wikipedia <dump>` nettoie le
      wikitexte (modèles imbriqués, tableaux, liens, références, catégories),
      écrit le JSONL attendu et consigne la source. Lecture en flux, jamais en
      mémoire. 44 tests, dont la préservation des élisions, des ligatures et des
      espaces insécables du français
- [ ] Lancer la conversion sur le vrai dump francophone et mesurer le volume
      réel en tokens — l'estimation « à peu près la bonne taille pour futo-mac »
      n'est pas vérifiée
- [x] **Licence des poids tranchée** (29/07) : CC BY-SA 4.0, comme les données
      de Wikipédia. Usage commercial permis, attribution et partage à
      l'identique obligatoires. Le code reste en Apache 2.0
- [ ] Choisir les autres sources et **vérifier chaque identifiant et chaque
      licence** (mentions [À VÉRIFIER] dans [docs/DONNEES.md](docs/DONNEES.md))
- [ ] Tenir `data/SOURCES.md` dès le premier téléchargement — exigé par le
      règlement européen sur l'IA, et impossible à reconstituer après coup
- [ ] Écrire le pipeline de filtrage, avec des seuils **ré-étalonnés sur le
      français** (les seuils publiés sont calibrés pour l'anglais et vident
      silencieusement un corpus français)
- [ ] Déduplication exacte, puis approchée si le volume le justifie
- [ ] Décontamination : retirer du corpus les phrases de
      `data/sondes/paires-minimales-fr.jsonl`, sinon la mesure grammaticale ne
      vaut plus rien
- [ ] Objectif intermédiaire : **10 G tokens** de français propre, documentés

## ⬜ Ensuite — le premier vrai modèle

- [x] **Tokenizer définitif entraîné** (30/07) : 2 Go du dump francophone réel,
      vocabulaire 32 768, 10 minutes sur Apple M2 Max. Le premier essai avait été
      jeté : `_lire_textes` donnait les lignes JSONL brutes au trainer, qui
      apprenait `{"text": "` et `", "titre": "` — des places prises sur 32 768,
      et du balisage que le modèle aurait recraché
- [x] **Fertilité mesurée** (30/07) : 1,561 token/mot · 4,45 octet/token ·
      73,4 % des mots rendus en un seul token, sur 29 771 mots. Les élisions
      tiennent : `L'`, ` qu'`, ` aujourd'` ressortent soudées, exactement ce que
      la découpe française visait
- [ ] Comparer cette fertilité à celle d'un tokenizer généraliste anglophone sur
      le MÊME texte français — c'est le chiffre publiable, et il justifie à lui
      seul le travail sur la découpe. Bloqué ici : HuggingFace est injoignable
      depuis l'environnement de développement, la comparaison doit se faire sur
      le Mac
- [ ] Entraîner `futo-small` (20-40 €) et publier les chiffres, bons ou mauvais
- [ ] Comparer 3 à 4 mélanges de corpus à cette échelle avant de passer à la suite
- [ ] Carte du modèle honnête : données, limites, biais, empreinte carbone
      (voir [docs/CARTE-DU-MODELE.md](docs/CARTE-DU-MODELE.md))
- [x] ~~Décider de la licence des poids avant l'entraînement~~ — fait le 29/07 :
      CC BY-SA 4.0

## 🎹 Piste produit — clavier à suggestions

Idée de Guillaume (30/07) : quatre cases au-dessus des touches, le bon mot
dedans. C'est exactement la tâche du modèle — prédire le mot suivant — mais la
perplexité n'y répond pas : elle note la probabilité du bon mot, jamais son
RANG, et un clavier ne montre que quatre rangs.

- [x] `futo eval` mesure désormais le taux de réussite top-1/3/4/5, sur tous les
      tokens ET sur les seuls débuts de mot. La seconde mesure est la seule
      honnête : compléter « aujourd' » par « hui » est facile, proposer le mot
      suivant quand rien n'est encore tapé l'est beaucoup moins
- [ ] Relever le chiffre sur un point de reprise de `futo-mac`
- [ ] Obstacle connu, et il est de fond : le modèle est entraîné sur Wikipédia.
      Un clavier sert à écrire des messages, pas des articles encyclopédiques.
      Le registre ne correspond pas, et aucune quantité d'entraînement ne
      corrigera cela — il faut du corpus conversationnel (sous-titres, forums)
- [ ] Autres travaux nécessaires avant tout produit : contrainte de préfixe
      (ne proposer que des mots compatibles avec ce qui est déjà tapé),
      poursuite des mots en plusieurs tokens, quantification pour tenir sur un
      téléphone (39 M de paramètres ≈ 79 Mo en fp16, 20 Mo en int4)

## ⬜ Plus tard

- [ ] `futo-base` (150-250 €), si `futo-small` tient ses promesses
- [ ] `futo-large` (1 500-2 500 €), seulement si tout le reste a tenu
- [ ] Réglage par instructions, et les tokens de dialogue déjà réservés dans le
      tokenizer (`<|reserve_0|>` à `<|reserve_11|>` — prévus pour n'avoir jamais
      à réentraîner le tokenizer, donc le modèle)
- [ ] Export vers d'autres écosystèmes (extra `[hf]`, volontairement isolé du cœur)
- [ ] Quantification pour l'inférence

## 🛠 Dette technique connue

- [ ] Le cache KV grandit par concaténation : simple et suffisant pour quelques
      centaines de tokens, à remplacer par un tampon préalloué pour de la
      production
- [ ] Le tirage des fenêtres se fait avec remise plutôt que par permutation
      exacte du corpus. Négligeable sur une seule époque, à revoir si l'on passe
      à plusieurs passes
- [x] ~~Le nettoyeur de wikitexte n'a jamais vu de vrai dump~~ — confronté le
      29/07 aux 1 000 premiers articles de frwiki. 6,6 % des documents portaient
      une trace de balisage ; trois défauts corrigés, tous invisibles sur des
      fragments bien formés : un bloc jamais refermé emportait la fin de
      l'article, modèles et tableaux imbriqués l'un dans l'autre étaient coupés,
      et une longue rafale d'apostrophes laissait des débris
- [x] ~~Refaire le contrôle sur 10 000 articles~~ — fait le 29/07 : verdict
      PROPRE, 0,02 % de documents portant une trace (4 occurrences dans 2
      documents), contre 6,6 % avant correction. Deux défauts de CONTENU
      trouvés à la lecture des extraits, invisibles pour les compteurs : les
      titres de sections consécutives ressortaient collés, et les modèles de
      date, de siècle, de nombre et d'unité étaient supprimés, ce qui laissait
      « né le à Moulins » — du français agrammatical
- [ ] Lancer la conversion COMPLÈTE et mesurer le volume réel en tokens.
      Sur 10 000 articles : 145 Mo, 40,7 M tokens, médiane 6 424 caractères —
      mais l'échantillon reste biaisé vers les articles les plus anciens, donc
      les plus longs. L'extrapolation à 2,7 millions d'articles n'est pas fiable
- [ ] `futo data telecharger` n'affiche qu'une marche à suivre pour les sources
      autres que Wikipédia — volontaire tant que leurs identifiants ne sont pas
      vérifiés
- [ ] Aucune mesure de MFU réelle sur GPU — les durées annoncées sont calculées,
      pas mesurées. À corriger au premier entraînement
- [x] ~~Le chemin Mac (MPS) n'a jamais été exécuté~~ — fait le 29/07 sur
      **Apple M2 Max**. Trois défauts trouvés, tous invisibles depuis une
      intégration continue sur processeur : (1) la reprise après plantage était
      cassée sur TOUT accélérateur, map_location déplaçant l'état du générateur
      aléatoire hors du processeur ; (2) le test de déterminisme exigeait le bit
      près, impossible sur MPS où l'ordre des réductions varie ; (3) les tests de
      documentation ratissaient les fichiers Markdown de .venv (239 tests au lieu
      de 188)
- [x] ~~Le débit de `futo-mac` sur Mac reste NON MESURÉ~~ — mesuré le 29/07 sur
      Apple M2 Max : 9 068 tokens/s, 19,1 % de MFU, 45,8 h pour le run complet.
      L'hypothèse de 15 % de MFU était conservatrice ; `futo info` est recalé
      sur 18 %, avec une marge
- [ ] Le MFU est rapporté à une crête bf16 quelle que soit la précision réelle :
      un run fp32 affiche donc un MFU environ deux fois trop bas

## 📏 Chiffres mesurés à ce jour

Tout ce qui suit vient d'une exécution réelle, pas d'une estimation.

| Mesure | Valeur | Conditions |
|---|---|---|
| **Fertilité du tokenizer définitif** | **1,561 token/mot · 4,45 octet/token** | **vocab 32 768, 2 Go de frwiki réel** |
| **Mots rendus en un seul token** | **73,4 %** | idem, sur 29 771 mots |
| Entraînement du tokenizer | 10 min | Apple M2 Max, 2 Go lus |
| Fertilité du tokenizer (ancien) | 1,62 token/mot · 3,69 octet/token | vocab 4 096, corpus d'exemple 142 Ko |
| **Rapport octets/token sur Wikipédia** | **3,56 octet/token** | mesuré sur 1 000 puis 10 000 articles réels |
| Extraction Wikipédia | 10 000 articles → 145 Mo, 40,7 M tokens | dump frwiki, médiane 6 424 car. |
| Propreté du corpus extrait | 0,02 % de documents avec une trace | après correction ; 6,6 % avant |
| Mots rendus en un seul token | 62,6 % | idem |
| Aller-retour tokenizer | exact | accents, ligatures, guillemets, émojis, code |
| Suite de tests | 279 tests, 17 s | 4 cœurs, hors ligne |
| Suite de tests | 14 s (190 tests à la date de la mesure) | Apple M2 Max |
| Débit `futo-tiny` | 57 000-60 000 tokens/s | **Apple M2 Max**, MPS, fp32 |
| Entraînement `futo-tiny` | 400 pas en 15,6 s | **Apple M2 Max**, MPS, fp32 |
| MFU `futo-tiny` | 4,0-4,2 % | M2 Max, fp32 contre crête bf16 — non représentatif |
| **Débit `futo-mac`** | **9 068 tokens/s** | **Apple M2 Max, MPS, bf16** |
| **MFU `futo-mac`** | **19,1 %** | idem — 2,63 TFLOPS effectifs |
| **Run complet `futo-mac`** | **45,8 h (1,9 j)** | idem, 1,49 G tokens |
| bfloat16 sur MPS | fonctionne | M2 Max, PyTorch 2.13 — aucun repli déclenché |
| Entraînement `futo-tiny` | 400 pas, 45-77 s, perte 8,3 → 4,17 | 4 cœurs, fp32 (deux mesures) |
| Débit `futo-tiny` | 11 000-19 000 tokens/s | 4 cœurs, fp32 |

Les chiffres qui manquent — et qui comptent — sont désormais ceux d'un vrai
entraînement de bout en bout : bits par octet sur du français tenu à l'écart,
score aux sondes grammaticales, fertilité du tokenizer sur un corpus réel. Le
débit, lui, est mesuré ; ce qui reste à prouver, c'est que le modèle apprend
quelque chose.

Aucun MFU n'a encore été relevé sur GPU NVIDIA : les durées et les coûts de
`futo-small`, `futo-base` et `futo-large` restent calculés.
