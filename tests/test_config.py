"""Tests du système de configuration.

Le point qui compte : une clé inconnue doit être une ERREUR. Une faute de frappe
dans un hyperparamètre qui passerait inaperçue coûterait un entraînement entier
mené avec la valeur par défaut, sans que rien ne le signale.
"""

from __future__ import annotations

import pytest

from futo.config import (
    FutoConfig,
    ModelConfig,
    appliquer_surcharges,
    charger_config,
)

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_cle_inconnue_est_une_erreur():
    with pytest.raises(ValueError, match="clé\\(s\\) inconnue\\(s\\)"):
        FutoConfig.from_dict({"model": {"n_layers": 12}})  # faute : n_layer


def test_le_message_liste_les_cles_acceptees():
    """Le message d'erreur doit aider à corriger, pas seulement constater."""
    with pytest.raises(ValueError, match="n_layer"):
        FutoConfig.from_dict({"model": {"nlayer": 12}})


def test_types_convertis_selon_lannotation():
    cfg = FutoConfig.from_dict({
        "model": {"n_layer": "8", "vocab_size": 1024},
        "train": {"lr": "3e-4", "compile": "true", "max_steps": 100,
                  "warmup_steps": 10},
    })
    assert cfg.model.n_layer == 8 and isinstance(cfg.model.n_layer, int)
    assert cfg.train.lr == pytest.approx(3e-4) and isinstance(cfg.train.lr, float)
    assert cfg.train.compile is True


@pytest.mark.parametrize("valeur,attendu", [
    ("true", True), ("vrai", True), ("oui", True), ("1", True), (1, True),
    ("false", False), ("faux", False), ("non", False), ("0", False), (0, False),
])
def test_booleens_acceptent_le_francais(valeur, attendu):
    cfg = FutoConfig.from_dict({"model": {"tie_weights": valeur}})
    assert cfg.model.tie_weights is attendu


def test_booleen_invalide_est_refuse():
    with pytest.raises(ValueError, match="booléen attendu"):
        FutoConfig.from_dict({"model": {"tie_weights": "peut-être"}})


def test_entier_non_entier_est_refuse():
    with pytest.raises(ValueError, match="entier attendu"):
        FutoConfig.from_dict({"model": {"n_layer": 3.5}})


def test_valeur_optionnelle_accepte_none():
    cfg = FutoConfig.from_dict({"train": {"decay_steps": None}})
    assert cfg.train.decay_steps is None


# --------------------------------------------------------------------------- #
# Surcharges en ligne de commande
# --------------------------------------------------------------------------- #


def test_surcharge_simple():
    brut = appliquer_surcharges({}, ["model.n_layer=16", "train.lr=1e-4"])
    assert brut["model"]["n_layer"] == 16
    assert brut["train"]["lr"] == pytest.approx(1e-4)


def test_surcharge_ecrase_le_yaml():
    brut = appliquer_surcharges({"model": {"n_layer": 12, "d_model": 768}},
                                ["model.n_layer=24"])
    assert brut["model"]["n_layer"] == 24
    assert brut["model"]["d_model"] == 768  # le reste est préservé


def test_surcharge_cree_les_niveaux_manquants():
    brut = appliquer_surcharges({}, ["train.dossier_sortie=/tmp/x"])
    assert brut["train"]["dossier_sortie"] == "/tmp/x"


def test_surcharge_mal_formee_est_refusee():
    with pytest.raises(ValueError, match="mal formée"):
        appliquer_surcharges({}, ["model.n_layer"])
    with pytest.raises(ValueError, match="chemin vide"):
        appliquer_surcharges({}, ["=12"])


def test_surcharge_texte_reste_du_texte():
    """Une valeur qui n'est pas du JSON valide doit être gardée en chaîne."""
    brut = appliquer_surcharges({}, ["nom=mon-essai"])
    assert brut["nom"] == "mon-essai"


# --------------------------------------------------------------------------- #
# Fichiers YAML et héritage
# --------------------------------------------------------------------------- #


def test_heritage_entre_configurations(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "model:\n  n_layer: 12\n  d_model: 768\ntrain:\n  lr: 6.0e-4\n",
        encoding="utf-8",
    )
    (tmp_path / "enfant.yaml").write_text(
        "hérite: base.yaml\nnom: enfant\nmodel:\n  n_layer: 24\n", encoding="utf-8"
    )
    cfg = charger_config(tmp_path / "enfant.yaml")
    assert cfg.nom == "enfant"
    assert cfg.model.n_layer == 24  # surchargé
    assert cfg.model.d_model == 768  # hérité
    assert cfg.train.lr == pytest.approx(6e-4)  # hérité


def test_heritage_circulaire_est_detecte(tmp_path):
    (tmp_path / "a.yaml").write_text("hérite: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("hérite: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circulaire"):
        charger_config(tmp_path / "a.yaml")


def test_config_introuvable(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        charger_config(tmp_path / "absent.yaml")


def test_toutes_les_configs_du_depot_sont_valides():
    """Chaque fichier livré doit se charger et être cohérent."""
    from pathlib import Path

    dossier = Path(__file__).resolve().parent.parent / "configs"
    fichiers = sorted(p for p in dossier.glob("futo-*.yaml"))
    assert fichiers, "Aucune configuration trouvée dans configs/."
    for chemin in fichiers:
        cfg = charger_config(chemin)
        assert cfg.model.nombre_parametres() > 0
        assert cfg.train.max_steps > cfg.train.warmup_steps
        assert cfg.model.n_head % cfg.model.n_kv_head == 0
        # Le vocabulaire doit tenir dans un uint16, sinon les shards doublent
        # de taille sans que personne ne s'en aperçoive.
        assert cfg.model.vocab_size <= 65_535, f"{chemin.name} : vocabulaire trop grand"


def test_configs_annoncent_leur_taille_reelle():
    """Les tailles écrites dans les commentaires doivent correspondre au calcul.

    Ces chiffres servent à décider d'une location GPU : ils ne doivent pas
    dériver silencieusement quand on modifie une configuration.
    """
    from pathlib import Path

    dossier = Path(__file__).resolve().parent.parent / "configs"
    attendus = {
        "futo-tiny": (1e6, 2e6),
        "futo-small": (95e6, 105e6),
        "futo-base": (290e6, 310e6),
        "futo-large": (1.15e9, 1.25e9),
    }
    for nom, (mini, maxi) in attendus.items():
        cfg = charger_config(dossier / f"{nom}.yaml")
        n = cfg.model.nombre_parametres()
        assert mini <= n <= maxi, (
            f"{nom} : {n / 1e6:.1f} M paramètres, hors de la fourchette annoncée "
            f"({mini / 1e6:.0f}–{maxi / 1e6:.0f} M)."
        )


# --------------------------------------------------------------------------- #
# Calculs dérivés
# --------------------------------------------------------------------------- #


def test_tokens_par_pas():
    cfg = FutoConfig.from_dict({
        "model": {"block_size": 1024},
        "data": {"batch_size": 16},
        "train": {"grad_accum": 32, "max_steps": 100, "warmup_steps": 10},
    })
    assert cfg.tokens_par_pas() == 16 * 1024 * 32
    assert cfg.tokens_par_pas(world_size=8) == 16 * 1024 * 32 * 8


def test_flops_par_token_croit_avec_la_taille():
    petit = ModelConfig(n_layer=6, d_model=512, n_head=8, n_kv_head=8)
    grand = ModelConfig(n_layer=12, d_model=512, n_head=8, n_kv_head=8)
    assert grand.flops_par_token() > petit.flops_par_token()


def test_flops_incluent_le_cout_de_lattention():
    """Doubler le contexte doit augmenter les FLOPs : l'attention est quadratique.

    Une formule qui se limiterait à 6·N raterait ce terme, et surestimerait le
    MFU des modèles à long contexte.
    """
    court = ModelConfig(block_size=512, n_layer=12, d_model=768, n_head=12, n_kv_head=4)
    long = ModelConfig(block_size=4096, n_layer=12, d_model=768, n_head=12, n_kv_head=4)
    assert long.flops_par_token() > court.flops_par_token()


def test_aller_retour_dictionnaire():
    """Une configuration doit survivre à un tour par le dictionnaire : c'est ce
    qui se passe à chaque sauvegarde de checkpoint."""
    original = charger_config(None, ["model.n_layer=7", "train.lr=1.5e-4"])
    copie = FutoConfig.from_dict(original.to_dict())
    assert copie.to_dict() == original.to_dict()


def test_resume_est_lisible():
    cfg = charger_config(None, ["nom=essai"])
    texte = cfg.resume()
    assert "essai" in texte
    assert "paramètres" in texte
    assert "tokens/pas" in texte
