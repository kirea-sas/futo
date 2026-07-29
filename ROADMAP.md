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
- [x] 190 tests, 18 s sur processeur, sans réseau — CI GitHub Actions,
      et le démarrage du README rejoué depuis un clone neuf dans un
      environnement virtuel vierge, jusqu'à la génération de texte,
      dont des tests de la DOCUMENTATION : les blocs de commandes doivent
      être copiables-collables (une apostrophe dans un commentaire shell
      bloque zsh sur « quote> »), les liens internes doivent pointer
      quelque part, et les sous-commandes citées doivent exister

## 🔥 Prochain chantier — les données

C'est **le** sujet. Le reste est de la plomberie déjà écrite.

- [ ] Choisir les sources et **vérifier chaque identifiant et chaque licence**
      (voir les mentions [À VÉRIFIER] dans [docs/DONNEES.md](docs/DONNEES.md))
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

- [ ] Entraîner le tokenizer définitif sur 2 Go du corpus réel, vocabulaire 32 768
- [ ] Mesurer sa fertilité et la comparer à celle d'un tokenizer généraliste
      anglophone sur le même texte français — c'est le premier chiffre publiable
      du projet, et il justifie à lui seul le travail sur la découpe
- [ ] Entraîner `futo-small` (20-40 €) et publier les chiffres, bons ou mauvais
- [ ] Comparer 3 à 4 mélanges de corpus à cette échelle avant de passer à la suite
- [ ] Carte du modèle honnête : données, limites, biais, empreinte carbone
      (voir [docs/CARTE-DU-MODELE.md](docs/CARTE-DU-MODELE.md))
- [ ] Décider de la licence des poids **avant** l'entraînement, en fonction des
      corpus retenus

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
- [ ] `futo data telecharger` n'affiche qu'une marche à suivre : c'est
      volontaire tant que les identifiants de corpus ne sont pas vérifiés, mais
      cela reste à automatiser une fois les sources arrêtées
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
| Fertilité du tokenizer | 1,62 token/mot · 3,69 octet/token | vocab 4 096, corpus d'exemple 142 Ko |
| Mots rendus en un seul token | 62,6 % | idem |
| Aller-retour tokenizer | exact | accents, ligatures, guillemets, émojis, code |
| Suite de tests | 190 tests, 18 s | 4 cœurs, hors ligne |
| Suite de tests | 190 tests, 14 s | Apple M2 Max |
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
