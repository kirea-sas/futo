"""Fixtures partagées.

Tout est minuscule et construit à la volée : la suite doit tourner sur un
processeur, hors ligne, en moins de deux minutes. Aucun test ne télécharge quoi
que ce soit — c'est une contrainte, pas un accident : l'intégration continue
n'a pas accès au réseau au-delà de l'installation des paquets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from futo.config import DataConfig, FutoConfig, ModelConfig, TrainConfig  # noqa: E402
from futo.data import preparer_corpus  # noqa: E402
from futo.model import Futo  # noqa: E402
from futo.tokenizer import entrainer_tokenizer  # noqa: E402

DOSSIER_ECHANTILLON = RACINE / "data" / "echantillon"
CHEMIN_SONDES = RACINE / "data" / "sondes" / "paires-minimales-fr.jsonl"


# --------------------------------------------------------------------------- #
# Texte
# --------------------------------------------------------------------------- #

# Un texte de secours, pour que les tests du tokenizer restent exécutables même
# si le corpus d'exemple venait à manquer. Il concentre volontairement les
# difficultés du français : élisions, deux apostrophes, ligatures, accents,
# traits d'union, guillemets, nombres.
TEXTE_SECOURS = """\
L'homme qu'il avait rencontré aujourd'hui n'était pas celui-ci.
Est-ce que vous croyez qu'elle viendra ? Peut-être, dit-il, si le temps s'y prête.
Le cœur de l'œuvre, c'est-à-dire l'essentiel, tient en 1 234,56 € et 12 % de patience.
« Va-t'en ! » lui cria-t-elle ; il s'en alla sans se retourner, l’air buté.
Les sœurs ont acheté des œufs, du bœuf et des vœux de bonne année à Nîmes.
Y a-t-il quelqu'un pour m'expliquer pourquoi, lorsqu'on écrit, jusqu'à présent tout va bien ?
Elle songea qu'il eût fallu partir plus tôt ; l’après-midi touchait à sa fin.
Donne-le-moi, va-t'en, celui-là, arc-en-ciel, porte-monnaie, presqu'île.
"""


@pytest.fixture(scope="session")
def texte_francais() -> str:
    """Du vrai français, tiré du corpus d'exemple si présent."""
    fichiers = sorted(DOSSIER_ECHANTILLON.glob("*.txt"))
    if fichiers:
        morceaux = [f.read_text(encoding="utf-8") for f in fichiers]
        return "\n\n".join(morceaux)
    return TEXTE_SECOURS * 200


@pytest.fixture(scope="session")
def texte_court() -> str:
    """Un texte court, suffisant pour les vérifications d'aller-retour."""
    return TEXTE_SECOURS


# --------------------------------------------------------------------------- #
# Tokenizer, shards, modèle
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory, texte_francais):
    """Un tokenizer réel, entraîné une seule fois pour toute la session."""
    dossier = tmp_path_factory.mktemp("tokenizer")
    source = dossier / "corpus.txt"
    source.write_text(texte_francais, encoding="utf-8")
    return entrainer_tokenizer(
        [source], dossier / "futo-tokenizer.json", vocab_size=2_048,
        frequence_min=2, verbeux=False,
    )


@pytest.fixture(scope="session")
def shards(tmp_path_factory, tokenizer, texte_francais):
    """Des shards préparés à partir du corpus d'exemple."""
    dossier = tmp_path_factory.mktemp("shards")
    source = dossier / "corpus.txt"
    source.write_text(texte_francais, encoding="utf-8")
    preparer_corpus(
        [source], tokenizer, dossier, tokens_par_shard=20_000,
        fraction_val=0.05, journal=None,
    )
    return dossier


@pytest.fixture
def config_modele() -> ModelConfig:
    """Un modèle jouet : assez petit pour être instantané, assez complet pour
    exercer GQA (4 têtes de requête pour 2 têtes clé/valeur)."""
    return ModelConfig(
        vocab_size=256, block_size=64, n_layer=2, n_head=4,
        n_kv_head=2, d_model=64, dropout=0.0,
    )


@pytest.fixture
def modele(config_modele) -> Futo:
    torch.manual_seed(0)
    m = Futo(config_modele)
    m.eval()  # pas de dropout, comportement déterministe
    return m


@pytest.fixture
def config_entrainement(tmp_path, shards, tokenizer) -> FutoConfig:
    """Une configuration d'entraînement complète, prête à tourner en secondes."""
    return FutoConfig(
        nom="test",
        model=ModelConfig(
            vocab_size=tokenizer.vocab_size, block_size=64, n_layer=2,
            n_head=4, n_kv_head=2, d_model=64, dropout=0.0,
        ),
        data=DataConfig(
            dossier=str(shards), tokenizer=str(tokenizer.chemin),
            batch_size=4, seed=99,
        ),
        train=TrainConfig(
            max_steps=6, warmup_steps=2, lr=1e-3, dtype="fp32",
            log_every=100, eval_every=0, save_every=0,
            dossier_sortie=str(tmp_path / "sortie"), seed=7,
        ),
    )
