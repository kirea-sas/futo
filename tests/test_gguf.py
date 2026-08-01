"""Tests de l'export GGUF.

Le format est lu par du code C qui ne pardonne rien : un décalage faux d'un
octet et le fichier est refusé, ou pire, accepté avec des poids décalés. Tout
est donc vérifié par relecture, jamais par inspection visuelle.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from futo.config import FutoConfig, ModelConfig
from futo.gguf import (
    ALIGNEMENT,
    MAGIE,
    VERSION,
    ecrire_gguf,
    exporter_gguf,
    lire_gguf,
    metadonnees_depuis_config,
    tenseurs_depuis_etat,
)
from futo.model import Futo


def _config(vocab: int = 64) -> FutoConfig:
    return FutoConfig(
        nom="essai",
        model=ModelConfig(
            vocab_size=vocab, n_layer=2, n_head=4, n_kv_head=2,
            d_model=32, block_size=16,
        ),
    )


def test_aller_retour_den_tete(tmp_path):
    chemin = ecrire_gguf(
        tmp_path / "a.gguf",
        {"general.architecture": "llama", "llama.block_count": 2, "essai.reel": 0.25},
        {"t.weight": np.zeros((3, 5), dtype=np.float16)},
    )
    assert chemin.read_bytes()[:4] == MAGIE

    relu = lire_gguf(chemin)
    assert relu["version"] == VERSION
    assert relu["metadonnees"]["general.architecture"] == "llama"
    assert relu["metadonnees"]["llama.block_count"] == 2
    assert relu["metadonnees"]["essai.reel"] == pytest.approx(0.25)
    # l'alignement est ajouté d'office : le lecteur en a besoin
    assert relu["metadonnees"]["general.alignment"] == ALIGNEMENT


def test_les_dimensions_sont_inscrites_a_lenvers():
    """Convention GGUF : la dimension qui varie le plus vite vient en premier.

    Une matrice PyTorch (sortie, entrée) doit ressortir (entrée, sortie). Se
    tromper ici donne un fichier que llama.cpp charge sans broncher et qui
    produit du bruit — le pire des cas.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as dossier:
        chemin = ecrire_gguf(
            Path(dossier) / "a.gguf", {},
            {"m.weight": np.zeros((7, 3), dtype=np.float32)},
        )
        assert lire_gguf(chemin)["tenseurs"]["m.weight"]["forme"] == [3, 7]


def test_les_donnees_sont_alignees(tmp_path):
    """Chaque tenseur doit commencer sur un multiple de l'alignement."""
    chemin = ecrire_gguf(
        tmp_path / "a.gguf", {},
        {
            "a.weight": np.zeros((3, 5), dtype=np.float16),   # 30 octets, non aligné
            "b.weight": np.zeros((2, 2), dtype=np.float32),
        },
    )
    tenseurs = lire_gguf(chemin)["tenseurs"]
    for nom, info in tenseurs.items():
        assert info["decalage"] % ALIGNEMENT == 0, f"{nom} mal aligné"


def test_les_octets_des_poids_survivent(tmp_path):
    """Ce qui est écrit doit pouvoir être relu à l'octet près."""
    attendu = np.arange(12, dtype=np.float32).reshape(4, 3).astype(np.float16)
    chemin = ecrire_gguf(tmp_path / "a.gguf", {}, {"t.weight": attendu})

    info = lire_gguf(chemin)["tenseurs"]["t.weight"]
    brut = chemin.read_bytes()
    # les données commencent au premier multiple de l'alignement après l'en-tête
    debut = len(brut) - (attendu.nbytes + (-attendu.nbytes % ALIGNEMENT))
    obtenu = np.frombuffer(brut[debut + info["decalage"]:], dtype=np.float16, count=12)
    assert np.array_equal(obtenu.reshape(4, 3), attendu)


def test_tous_les_tenseurs_du_modele_ont_un_nom_llama():
    """Aucun poids ne doit rester sur le carreau.

    Si le modèle gagne un tenseur et que la table de correspondance ne suit
    pas, l'export doit échouer bruyamment plutôt que produire un fichier
    incomplet.
    """
    modele = Futo(_config().model)
    tenseurs = tenseurs_depuis_etat(modele.state_dict())

    assert "token_embd.weight" in tenseurs
    assert "output_norm.weight" in tenseurs
    for couche in range(2):
        for suffixe in (
            "attn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
            "ffn_norm", "ffn_gate", "ffn_up", "ffn_down",
        ):
            assert f"blk.{couche}.{suffixe}.weight" in tenseurs
    # embeddings liés : output.weight n'est pas dupliqué, llama.cpp retombe
    # sur token_embd.weight — la moitié du fichier économisée
    assert "output.weight" not in tenseurs


def test_un_tenseur_inconnu_fait_echouer_lexport():
    with pytest.raises(KeyError, match="CORRESPONDANCE"):
        tenseurs_depuis_etat({"une.chose.inattendue": torch.zeros(2, 2)})


def test_les_metadonnees_decrivent_larchitecture_llama():
    cfg = _config()
    m = metadonnees_depuis_config(cfg)
    assert m["general.architecture"] == "llama"
    assert m["llama.block_count"] == cfg.model.n_layer
    assert m["llama.embedding_length"] == cfg.model.d_model
    assert m["llama.attention.head_count"] == cfg.model.n_head
    assert m["llama.attention.head_count_kv"] == cfg.model.n_kv_head
    assert m["llama.rope.dimension_count"] == cfg.model.head_dim
    assert m["llama.feed_forward_length"] == cfg.model.d_ff
    assert m["llama.vocab_size"] == cfg.model.vocab_size


def test_export_complet_depuis_un_point_de_reprise(tmp_path):
    """Le chemin réellement emprunté : un checkpoint sur le disque, un GGUF."""
    cfg = _config()
    modele = Futo(cfg.model)
    checkpoint = tmp_path / "dernier.pt"
    torch.save({"config_objet": cfg, "modele": modele.state_dict(), "pas": 42}, checkpoint)

    sortie = exporter_gguf(checkpoint, tmp_path / "futo.gguf", nom="futo-essai")
    relu = lire_gguf(sortie)

    assert relu["metadonnees"]["general.name"] == "futo-essai"
    assert relu["metadonnees"]["general.file_type"] == 1  # F16
    assert len(relu["tenseurs"]) == 2 + 2 * 9  # embeddings, norme finale, 2 blocs
    assert relu["tenseurs"]["token_embd.weight"]["forme"] == [
        cfg.model.d_model, cfg.model.vocab_size
    ]


def test_export_en_pleine_precision(tmp_path):
    cfg = _config()
    modele = Futo(cfg.model)
    checkpoint = tmp_path / "dernier.pt"
    torch.save({"config_objet": cfg, "modele": modele.state_dict(), "pas": 1}, checkpoint)

    sortie = exporter_gguf(checkpoint, tmp_path / "f32.gguf", demi_precision=False)
    relu = lire_gguf(sortie)
    assert relu["metadonnees"]["general.file_type"] == 0
    assert relu["tenseurs"]["token_embd.weight"]["type"] == 0  # F32


def test_les_prefixes_de_compilation_sont_retires(tmp_path):
    """torch.compile et DDP préfixent les noms. Un export doit les ignorer."""
    cfg = _config()
    modele = Futo(cfg.model)
    etat = {f"_orig_mod.{c}": v for c, v in modele.state_dict().items()}
    checkpoint = tmp_path / "compile.pt"
    torch.save({"config_objet": cfg, "modele": etat, "pas": 1}, checkpoint)

    relu = lire_gguf(exporter_gguf(checkpoint, tmp_path / "a.gguf"))
    assert "token_embd.weight" in relu["tenseurs"]


def test_la_quantification_nest_pas_faite_ici(tmp_path):
    """Un tenseur entier doit être refusé, avec le bon conseil dans le message."""
    with pytest.raises(TypeError, match="llama-quantize"):
        ecrire_gguf(tmp_path / "a.gguf", {}, {"t": np.zeros((2, 2), dtype=np.int8)})


def test_les_poids_lies_ne_sont_pas_ecrits_deux_fois(tmp_path):
    """Sur futo-mac, dupliquer les embeddings ajouterait 33 Mo pour rien.

    llama.cpp retombe sur token_embd.weight quand output.weight manque : c'est
    exactement le comportement voulu. Un modèle SANS poids liés, lui, doit bien
    voir son output.weight exporté.
    """
    cfg = _config()
    cfg.model.tie_weights = True
    lie = Futo(cfg.model)
    assert "output.weight" not in tenseurs_depuis_etat(lie.state_dict())

    cfg_libre = _config()
    cfg_libre.model.tie_weights = False
    libre = Futo(cfg_libre.model)
    tenseurs = tenseurs_depuis_etat(libre.state_dict())
    assert "output.weight" in tenseurs, "sans poids liés, la tête doit être exportée"
    assert "token_embd.weight" in tenseurs


def test_un_tokenizer_qui_ne_correspond_pas_est_refuse(tmp_path):
    """Un vocabulaire de taille différente donne un GGUF que llama.cpp rejette.

    Autant le dire ici, avec la marche à suivre, plutôt que de laisser
    découvrir la panne au chargement sur un téléphone.
    """
    from futo.tokenizer import entrainer_tokenizer

    corpus = tmp_path / "c.txt"
    corpus.write_text("Le chat dort sur le tapis. " * 300, encoding="utf-8")
    tok = entrainer_tokenizer([corpus], tmp_path / "tok.json", vocab_size=400, verbeux=False)

    cfg = _config(vocab=tok.vocab_size + 7)  # décalage volontaire
    cfg.data.tokenizer = str(tmp_path / "tok.json")
    checkpoint = tmp_path / "dernier.pt"
    torch.save({"config_objet": cfg, "modele": Futo(cfg.model).state_dict()}, checkpoint)

    with pytest.raises(ValueError, match="refusé au chargement"):
        exporter_gguf(checkpoint, tmp_path / "a.gguf", tokenizer=tok)

    # sans tokenizer, l'export passe : le fichier ne porte alors aucun vocabulaire
    relu = lire_gguf(exporter_gguf(checkpoint, tmp_path / "b.gguf"))
    assert "tokenizer.ggml.tokens" not in relu["metadonnees"]


def test_les_reserves_suffisent_aux_jetons_du_clavier():
    """Le compte doit tenir, sinon c'est un réentraînement complet.

    Trente jetons sont exigés par le moteur du clavier ; le tokenizer en
    réserve quarante-deux. Si quelqu'un rabaisse RESERVES, ce test le dit.
    """
    from futo.gguf import JETONS_CLAVIER
    from futo.tokenizer import RESERVES

    assert len(JETONS_CLAVIER) == 30, "26 jetons de caractère + 4 de contrôle"
    assert RESERVES >= len(JETONS_CLAVIER), (
        f"{RESERVES} réserves pour {len(JETONS_CLAVIER)} jetons exigés : "
        f"il faudrait réentraîner le tokenizer ET le modèle."
    )


def test_les_jetons_de_caractere_restent_contigus():
    """Leur code fait LETTERS_TO_IDS[i] = LETTERS_TO_IDS[0] + i.

    Un trou dans la suite <CHAR_A>…<CHAR_Z> et le clavier lit n importe quoi,
    sans erreur visible.
    """
    from futo.gguf import JETONS_CLAVIER, renommer_reserves

    jetons = ["<|fin_de_texte|>"] + [f"<|reserve_{i}|>" for i in range(42)] + ["Ġchat"]
    renommes = renommer_reserves(jetons, JETONS_CLAVIER)

    positions = [renommes.index(f"<CHAR_{chr(ord('A') + i)}>") for i in range(26)]
    assert positions == list(range(positions[0], positions[0] + 26)), (
        f"jetons de caractère non contigus : {positions}"
    )
    # le reste du vocabulaire n a pas bougé
    assert renommes[0] == "<|fin_de_texte|>"
    assert renommes[-1] == "Ġchat"
    assert len(renommes) == len(jetons)


def test_un_vocabulaire_sans_assez_de_reserves_est_refuse():
    """Un tokenizer entraîné avant le passage à 42 réserves doit être signalé."""
    from futo.gguf import JETONS_CLAVIER, renommer_reserves

    ancien = ["<|fin_de_texte|>"] + [f"<|reserve_{i}|>" for i in range(12)]
    with pytest.raises(ValueError, match="réentraîner"):
        renommer_reserves(ancien, JETONS_CLAVIER)


def test_export_avec_les_jetons_du_clavier(tmp_path):
    """Le chemin complet : un checkpoint, un GGUF portant les jetons du clavier."""
    from futo.tokenizer import entrainer_tokenizer

    corpus = tmp_path / "c.txt"
    corpus.write_text("Le chat dort sur le tapis. L'homme qu'elle attend. " * 300,
                      encoding="utf-8")
    tok = entrainer_tokenizer([corpus], tmp_path / "tok.json", vocab_size=400, verbeux=False)

    cfg = _config(vocab=tok.vocab_size)
    checkpoint = tmp_path / "dernier.pt"
    torch.save({"config_objet": cfg, "modele": Futo(cfg.model).state_dict()}, checkpoint)

    sortie = exporter_gguf(checkpoint, tmp_path / "clavier.gguf",
                           tokenizer=tok, jetons_clavier=True)
    jetons = lire_gguf(sortie)["metadonnees"]["tokenizer.ggml.tokens"]

    for attendu in ("<CHAR_A>", "<CHAR_Z>", "<XBU>", "<XBC>", "<XEC>", "<XC0>"):
        assert attendu in jetons, f"{attendu} absent du GGUF"
    # sans le drapeau, rien n est renommé : aucun nom tiers dans l artefact
    nu = lire_gguf(exporter_gguf(checkpoint, tmp_path / "nu.gguf", tokenizer=tok))
    assert "<CHAR_A>" not in nu["metadonnees"]["tokenizer.ggml.tokens"]
    assert "<|reserve_0|>" in nu["metadonnees"]["tokenizer.ggml.tokens"]
