#!/usr/bin/env bash
#
# Vérifie la chaîne complète : tokenizer, données, contrôle, entraînement,
# reprise, évaluation, génération — sur le corpus d'exemple, hors ligne.
#
# Ce script est la SEULE définition de cette vérification. L'intégration
# continue l'appelle tel quel, et on peut le lancer à l'identique en local.
#
# C'est né d'une bêtise : les commandes de la CI étaient écrites directement
# dans le fichier YAML du workflow, donc jamais exécutées avant d'être poussées.
# L'une d'elles fixait max_steps=20 alors que la configuration a warmup_steps=20,
# ce que la validation refuse — et onze exécutions ont échoué d'affilée, chacune
# envoyant un courriel, pour une faute qui se voyait en une seconde en local.
#
# Règle qui en découle : aucune commande ne va dans un fichier de CI sans
# passer par ici d'abord.

set -euo pipefail

echo "═══ nettoyage des sorties précédentes ═══"
rm -rf data/tokenizer data/prepare sorties

echo
echo "═══ 1. tokenizer ═══"
futo tokenizer entrainer --corpus 'data/echantillon/*.txt' --vocab 2048 --silencieux

echo
echo "═══ 2. préparation des données ═══"
futo data preparer --corpus 'data/echantillon/*.txt' \
                   --fraction-val 0.05 --tokens-par-shard 20000 --silencieux

echo
echo "═══ 3. contrôle du corpus ═══"
futo data controler 'data/echantillon/*.txt' --echantillon 0

echo
echo "═══ 4. mesure de débit ═══"
futo bench configs/futo-tiny.yaml --micro-lots 3 --echauffement 1 \
           --set model.vocab_size=2048

echo
echo "═══ 5. entraînement ═══"
# max_steps doit rester STRICTEMENT supérieur à warmup_steps (20 dans
# futo-tiny.yaml) : on abaisse l'échauffement plutôt que de frôler la limite.
futo train configs/futo-tiny.yaml \
           --set model.vocab_size=2048 \
           --set train.max_steps=30 \
           --set train.warmup_steps=5 \
           --set train.eval_every=15 \
           --set train.save_every=15 \
           --silencieux

echo
echo "═══ 6. reprise depuis le dernier checkpoint ═══"
futo train configs/futo-tiny.yaml \
           --set model.vocab_size=2048 \
           --set train.max_steps=30 \
           --set train.warmup_steps=5 \
           --set train.eval_every=15 \
           --set train.save_every=15 \
           --reprendre auto --silencieux

echo
echo "═══ 7. évaluation ═══"
futo eval sorties/tiny/dernier.pt --lots 2

echo
echo "═══ 8. génération ═══"
futo generer sorties/tiny/dernier.pt --amorce "Le vieux moulin" --max-tokens 20 --seed 1

echo
echo "═══ 9. description des configurations ═══"
for config in configs/futo-*.yaml; do
  futo info "$config" > /dev/null
  echo "  $config décrite sans erreur"
done

echo
echo "✓ chaîne complète vérifiée"
