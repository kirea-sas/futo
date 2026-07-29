"""Tests du format de shards et du chargeur.

Deux propriétés comptent ici : aucun token ne doit être perdu ni dupliqué entre
le texte et les shards, et le chargeur doit être parfaitement reproductible —
c'est lui qui rend la reprise après plantage exacte.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from futo.data import (
    MAGIQUE,
    TAILLE_ENTETE,
    VERSION_FORMAT,
    ChargeurTokens,
    ecrire_shard,
    lire_documents,
    lire_entete,
    ouvrir_shard,
    preparer_corpus,
)


# --------------------------------------------------------------------------- #
# Format binaire
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_aller_retour_du_shard(tmp_path, dtype):
    tokens = np.arange(1000, dtype=dtype)
    chemin = tmp_path / "s.bin"
    ecrire_shard(chemin, tokens, vocab_size=2000)

    relu, entete = ouvrir_shard(chemin)
    assert entete.version == VERSION_FORMAT
    assert entete.n_tokens == 1000
    assert entete.vocab_size == 2000
    assert entete.dtype == np.dtype(dtype)
    assert np.array_equal(np.asarray(relu), tokens)


def test_taille_du_fichier_est_exacte(tmp_path):
    """En-tête de 1 024 octets, puis les tokens, et rien d'autre."""
    tokens = np.arange(500, dtype=np.uint16)
    chemin = tmp_path / "s.bin"
    ecrire_shard(chemin, tokens, vocab_size=1000)
    assert chemin.stat().st_size == TAILLE_ENTETE + 500 * 2


def test_fichier_etranger_est_rejete(tmp_path):
    """Un fichier qui n'est pas un shard doit être refusé, pas mal interprété."""
    chemin = tmp_path / "pas-un-shard.bin"
    chemin.write_bytes(b"\x00" * 4096)
    with pytest.raises(ValueError, match="nombre magique"):
        lire_entete(chemin)


def test_fichier_tronque_est_rejete(tmp_path):
    chemin = tmp_path / "tronque.bin"
    chemin.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="tronqué"):
        lire_entete(chemin)


def test_version_future_est_rejetee(tmp_path):
    """Un shard d'une version inconnue doit produire un message actionnable."""
    chemin = tmp_path / "futur.bin"
    entete = np.zeros(256, dtype=np.int32)
    entete[0], entete[1], entete[2], entete[3] = MAGIQUE, 99, 1, 0
    chemin.write_bytes(entete.astype("<i4").tobytes())
    with pytest.raises(ValueError, match="version 99"):
        lire_entete(chemin)


def test_token_hors_vocabulaire_est_refuse_a_lecriture(tmp_path):
    """Écrire un token ≥ vocab_size est une erreur de programmation : on l'attrape."""
    with pytest.raises(ValueError, match="hors du vocabulaire"):
        ecrire_shard(tmp_path / "s.bin", np.array([5, 300], dtype=np.uint16), vocab_size=100)


# --------------------------------------------------------------------------- #
# Préparation : rien ne doit se perdre
# --------------------------------------------------------------------------- #


def test_aucun_token_perdu_ni_duplique(tmp_path, tokenizer):
    """Les shards doivent contenir exactement les tokens des documents, dans l'ordre.

    C'est le test qui garantit qu'on entraîne bien sur le texte qu'on croit :
    une erreur de découpe ou de tampon passerait sinon totalement inaperçue.
    """
    documents = [f"Document numéro {i}, avec l'accent aigu et un œuf." for i in range(40)]
    source = tmp_path / "corpus.txt"
    source.write_text("\n\n".join(documents), encoding="utf-8")

    preparer_corpus(
        [source], tokenizer, tmp_path / "prep", tokens_par_shard=97,
        fraction_val=0.0, journal=None,
    )

    # Ce qu'on attend : chaque document encodé, suivi du token de fin.
    attendu: list[int] = []
    for doc in lire_documents(source, "ligne-vide"):
        attendu.extend(tokenizer.encoder(doc, ajouter_fin=True))

    obtenu: list[int] = []
    for chemin in sorted((tmp_path / "prep").glob("train_*.bin")):
        tableau, _ = ouvrir_shard(chemin)
        obtenu.extend(int(x) for x in tableau)

    assert obtenu == attendu, "Les shards ne correspondent pas au texte encodé."


def test_decoupe_en_shards_respecte_la_taille(tmp_path, tokenizer, texte_francais):
    preparer_corpus(
        [_ecrire(tmp_path, texte_francais)], tokenizer, tmp_path / "prep",
        tokens_par_shard=500, fraction_val=0.0, journal=None,
    )
    chemins = sorted((tmp_path / "prep").glob("train_*.bin"))
    assert len(chemins) > 1, "Le corpus aurait dû produire plusieurs shards."
    for chemin in chemins[:-1]:
        assert lire_entete(chemin).n_tokens == 500  # tous pleins sauf le dernier
    assert lire_entete(chemins[-1]).n_tokens <= 500


def test_separation_validation(tmp_path, tokenizer, texte_francais):
    resultat = preparer_corpus(
        [_ecrire(tmp_path, texte_francais)], tokenizer, tmp_path / "prep",
        tokens_par_shard=100_000, fraction_val=0.05, journal=None,
    )
    assert resultat.n_shards_val == 1
    assert resultat.n_tokens_val > 0
    # La validation ne doit pas dévorer tout le corpus.
    assert resultat.n_tokens_val < resultat.n_tokens / 2


def test_format_jsonl(tmp_path, tokenizer):
    import json

    source = tmp_path / "c.jsonl"
    source.write_text(
        "\n".join(json.dumps({"text": f"Texte {i} : l'été."}, ensure_ascii=False)
                  for i in range(30)),
        encoding="utf-8",
    )
    resultat = preparer_corpus(
        [source], tokenizer, tmp_path / "prep", tokens_par_shard=10_000,
        fraction_val=0.0, separateur="jsonl", journal=None,
    )
    assert resultat.n_documents == 30


def test_jsonl_sans_champ_texte_est_signale(tmp_path, tokenizer):
    source = tmp_path / "c.jsonl"
    source.write_text('{"autre": "valeur"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="aucun champ de texte"):
        preparer_corpus([source], tokenizer, tmp_path / "p", separateur="jsonl", journal=None)


def test_corpus_vide_est_signale(tmp_path, tokenizer):
    source = tmp_path / "vide.txt"
    source.write_text("   \n\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="Aucun shard"):
        preparer_corpus([source], tokenizer, tmp_path / "p", journal=None)


# --------------------------------------------------------------------------- #
# Chargeur
# --------------------------------------------------------------------------- #


def test_lots_deterministes(shards):
    """Le même pas doit toujours donner le même lot. C'est la base de la reprise."""
    a = ChargeurTokens(shards, "train", 32, 4, seed=5)
    b = ChargeurTokens(shards, "train", 32, 4, seed=5)
    for pas in (0, 1, 17, 500):
        x1, y1 = a.lot(pas)
        x2, y2 = b.lot(pas)
        assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_pas_differents_donnent_lots_differents(shards):
    chargeur = ChargeurTokens(shards, "train", 32, 4, seed=5)
    assert not torch.equal(chargeur.lot(0)[0], chargeur.lot(1)[0])


def test_graines_differentes_donnent_lots_differents(shards):
    a = ChargeurTokens(shards, "train", 32, 4, seed=1)
    b = ChargeurTokens(shards, "train", 32, 4, seed=2)
    assert not torch.equal(a.lot(0)[0], b.lot(0)[0])


def test_rangs_differents_voient_des_donnees_differentes(shards):
    """En multi-GPU, deux rangs ne doivent pas s'entraîner sur le même texte."""
    a = ChargeurTokens(shards, "train", 32, 4, seed=5, rang=0, monde=2)
    b = ChargeurTokens(shards, "train", 32, 4, seed=5, rang=1, monde=2)
    assert not torch.equal(a.lot(0)[0], b.lot(0)[0])


def test_cible_est_lentree_decalee_dun_token(shards):
    """La cible du pas t est l'entrée du pas t+1 : c'est la définition de la tâche."""
    chargeur = ChargeurTokens(shards, "train", 32, 4, seed=5)
    entree, cible = chargeur.lot(0)
    assert torch.equal(entree[:, 1:], cible[:, :-1])


def test_formes_et_types_du_lot(shards):
    chargeur = ChargeurTokens(shards, "train", 32, 6, seed=5)
    entree, cible = chargeur.lot(0)
    assert entree.shape == cible.shape == (6, 32)
    # int64 : c'est ce qu'attend nn.Embedding, et uint16 n'existe pas côté PyTorch.
    assert entree.dtype == torch.int64


def test_tokens_dans_le_vocabulaire(shards):
    chargeur = ChargeurTokens(shards, "train", 32, 8, seed=5)
    entree, cible = chargeur.lot(0)
    assert int(entree.max()) < chargeur.vocab_size
    assert int(cible.max()) < chargeur.vocab_size
    assert int(entree.min()) >= 0


def test_dossier_sans_shard_leve_une_erreur_actionnable(tmp_path):
    with pytest.raises(FileNotFoundError, match="futo data preparer"):
        ChargeurTokens(tmp_path, "train", 32, 4)


def test_shard_plus_court_que_le_contexte_est_signale(tmp_path):
    """Mieux vaut refuser bruyamment que sauter un shard en silence."""
    ecrire_shard(tmp_path / "train_00000.bin", np.arange(10, dtype=np.uint16), 100)
    with pytest.raises(ValueError, match="moins que le contexte"):
        ChargeurTokens(tmp_path, "train", 64, 2)


def test_shards_de_vocabulaires_differents_sont_refuses(tmp_path):
    """Mélanger des shards de deux tokenizers donnerait un modèle incohérent."""
    ecrire_shard(tmp_path / "train_00000.bin", np.arange(200, dtype=np.uint16), 300)
    ecrire_shard(tmp_path / "train_00001.bin", np.arange(200, dtype=np.uint16), 400)
    with pytest.raises(ValueError, match="plusieurs tailles de"):
        ChargeurTokens(tmp_path, "train", 32, 2)


def _ecrire(dossier, texte):
    chemin = dossier / "corpus.txt"
    chemin.write_text(texte, encoding="utf-8")
    return chemin
