"""Tests de la boucle d'entraînement.

Deux d'entre eux valent tous les autres :

* `test_surapprentissage_dun_lot` prouve que le modèle apprend vraiment. Sans
  lui, on peut avoir un code qui tourne des heures sans jamais rien apprendre —
  gradients détachés, cibles mal alignées, taux d'apprentissage nul.
* `test_reprise_donne_exactement_la_meme_suite` prouve qu'un plantage ne coûte
  rien. C'est ce qui protège plusieurs centaines d'euros de location GPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from futo.config import FutoConfig, ModelConfig, TrainConfig
from futo.data import ChargeurTokens
from futo.model import Futo
from futo.train import charger, entrainer, sauvegarder, taux_apprentissage


# --------------------------------------------------------------------------- #
# Est-ce que ça apprend ?
# --------------------------------------------------------------------------- #


def test_surapprentissage_dun_lot():
    """Sur un unique lot répété, la perte doit s'effondrer.

    Un modèle correct mémorise trivialement quelques dizaines de tokens. Si la
    perte reste bloquée près de ln(V), c'est que le gradient ne circule pas.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2,
                      n_kv_head=1, d_model=64, dropout=0.0)
    modele = Futo(cfg)
    modele.train()

    entree = torch.randint(0, cfg.vocab_size, (4, 16))
    cible = torch.randint(0, cfg.vocab_size, (4, 16))

    optimiseur = torch.optim.AdamW(modele.groupes_parametres(0.0), lr=3e-3)
    perte_initiale = None
    for _ in range(220):
        optimiseur.zero_grad(set_to_none=True)
        _, perte = modele(entree, cibles=cible)
        perte.backward()
        optimiseur.step()
        if perte_initiale is None:
            perte_initiale = perte.item()

    perte_finale = perte.item()
    assert perte_initiale > 3.0, f"Perte initiale suspecte : {perte_initiale:.3f}"
    assert perte_finale < 0.15, (
        f"Le modèle n'apprend pas : la perte est passée de {perte_initiale:.3f} "
        f"à {perte_finale:.3f} seulement."
    )


def test_tous_les_parametres_recoivent_un_gradient():
    """Un paramètre sans gradient est du poids mort : souvent un oubli de câblage."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=4,
                      n_kv_head=2, d_model=64)
    modele = Futo(cfg)
    modele.train()
    entree = torch.randint(0, cfg.vocab_size, (2, 16))
    _, perte = modele(entree, cibles=entree)
    perte.backward()

    orphelins = [
        nom for nom, p in modele.named_parameters()
        if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))
    ]
    assert not orphelins, f"Paramètres sans gradient : {orphelins}"


def test_lentrainement_complet_fait_baisser_la_perte(config_entrainement):
    """La chaîne complète, de bout en bout, sur de vraies données françaises."""
    config_entrainement.train.max_steps = 30
    config_entrainement.train.lr = 3e-3
    resultat = entrainer(config_entrainement, verbeux=False)

    assert resultat.pas_effectues == 30
    assert math.isfinite(resultat.perte_finale)
    assert (resultat.dossier / "dernier.pt").exists()
    assert (resultat.dossier / "journal.jsonl").exists()
    assert (resultat.dossier / "journal.csv").exists()

    # Le journal doit contenir des mesures exploitables.
    import json

    lignes = [
        json.loads(l)
        for l in (resultat.dossier / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert lignes and "perte" in lignes[0]
    pertes = [l["perte"] for l in lignes if "perte" in l]
    assert pertes[-1] < pertes[0], (
        f"La perte n'a pas baissé : {pertes[0]:.3f} → {pertes[-1]:.3f}"
    )


# --------------------------------------------------------------------------- #
# Reprise exacte
# --------------------------------------------------------------------------- #


def test_reprise_donne_exactement_la_meme_suite(config_entrainement, tmp_path):
    """Repartir d'un checkpoint intermédiaire doit rejouer la suite à l'identique.

    C'est la propriété qui rend un plantage indolore, et elle repose sur trois
    choses : le chargeur sans état, la sauvegarde de l'état de l'optimiseur, et
    celle des générateurs aléatoires.

    Le protocole reproduit le cas réel : une seule et même configuration —
    donc le même planning de taux d'apprentissage —, un entraînement de 8 pas
    qui laisse un checkpoint au pas 4, puis une reprise depuis ce checkpoint.
    Les poids finaux doivent coïncider au bit près.

    (Faire varier `max_steps` entre les deux moitiés, comme on serait tenté de
    l'écrire, changerait la forme du cosinus et donc le taux d'apprentissage :
    le test échouerait pour une raison sans rapport avec la reprise.)
    """
    reference = _copier(config_entrainement, tmp_path / "reference")
    reference.train.max_steps = 8
    reference.train.save_every = 4  # laisse un checkpoint au pas 4
    entrainer(reference, verbeux=False)
    modele_reference, _ = charger(reference.train.dossier_sortie + "/dernier.pt")

    intermediaire = tmp_path / "reference" / "pas_0000004.pt"
    assert intermediaire.exists(), "Le checkpoint intermédiaire n'a pas été écrit."

    suite = _copier(config_entrainement, tmp_path / "suite")
    suite.train.max_steps = 8
    suite.train.save_every = 4
    suite.train.reprendre = str(intermediaire)
    entrainer(suite, verbeux=False)
    modele_repris, _ = charger(suite.train.dossier_sortie + "/dernier.pt")

    repris = dict(modele_repris.named_parameters())
    ecarts = {
        nom: (p - repris[nom]).abs().max().item()
        for nom, p in modele_reference.named_parameters()
    }
    pire = max(ecarts.values())
    assert pire < 1e-6, (
        f"La reprise ne redonne pas les mêmes poids (écart max {pire:.2e} sur "
        f"{max(ecarts, key=ecarts.get)})."
    )


def test_checkpoint_est_autoportant(config_entrainement):
    """Un checkpoint doit se recharger sans le YAML d'origine."""
    config_entrainement.train.max_steps = 3
    resultat = entrainer(config_entrainement, verbeux=False)

    modele, charge = charger(resultat.dossier / "dernier.pt")
    assert charge["pas"] == 3
    assert isinstance(charge["config_objet"], FutoConfig)
    assert modele.cfg.n_layer == config_entrainement.model.n_layer
    assert modele.cfg.vocab_size == config_entrainement.model.vocab_size

    entree = torch.randint(0, modele.cfg.vocab_size, (1, 8))
    logits, _ = modele(entree, cibles=entree)
    assert torch.isfinite(logits).all()


def test_sauvegarde_est_atomique(tmp_path, config_modele):
    """Aucun fichier temporaire ne doit subsister après une sauvegarde réussie."""
    modele = Futo(config_modele)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=1e-3)
    cfg = FutoConfig(model=config_modele)
    chemin = tmp_path / "c.pt"
    sauvegarder(chemin, modele, optimiseur, cfg, pas=1, meilleure_val=1.0)
    assert chemin.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_absent_leve_une_erreur_claire(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        charger(tmp_path / "absent.pt")


def test_nettoyage_garde_les_plus_recents(config_entrainement):
    """Les vieux checkpoints périodiques sont supprimés, pas « dernier » ni « meilleur »."""
    config_entrainement.train.max_steps = 10
    config_entrainement.train.save_every = 2
    config_entrainement.train.garder_n_checkpoints = 2
    resultat = entrainer(config_entrainement, verbeux=False)

    periodiques = sorted(resultat.dossier.glob("pas_*.pt"))
    assert len(periodiques) == 2, f"Attendu 2 checkpoints, trouvé {len(periodiques)}"
    assert (resultat.dossier / "dernier.pt").exists()


# --------------------------------------------------------------------------- #
# Déterminisme et cohérence
# --------------------------------------------------------------------------- #


def test_meme_graine_meme_resultat(config_entrainement, tmp_path):
    a = _copier(config_entrainement, tmp_path / "a")
    b = _copier(config_entrainement, tmp_path / "b")
    r1 = entrainer(a, verbeux=False)
    r2 = entrainer(b, verbeux=False)
    assert abs(r1.perte_finale - r2.perte_finale) < 1e-9


def test_graine_differente_resultat_different(config_entrainement, tmp_path):
    a = _copier(config_entrainement, tmp_path / "a")
    b = _copier(config_entrainement, tmp_path / "b")
    b.train.seed = a.train.seed + 1
    b.data.seed = a.data.seed + 1
    assert abs(entrainer(a, verbeux=False).perte_finale
               - entrainer(b, verbeux=False).perte_finale) > 1e-9


def test_incoherence_de_vocabulaire_est_detectee(config_entrainement):
    """Entraîner avec un vocabulaire différent de celui des shards doit échouer
    tout de suite, avec un message qui dit quoi corriger."""
    config_entrainement.model.vocab_size += 1
    with pytest.raises(ValueError, match="Incohérence de vocabulaire"):
        entrainer(config_entrainement, verbeux=False)


def test_accumulation_de_gradient_equivaut_a_un_grand_lot(config_entrainement, tmp_path):
    """4 micro-lots de 2 doivent donner la même perte moyenne qu'un lot de 8.

    Pas les mêmes poids — les lots tirés diffèrent — mais un ordre de grandeur
    identique, ce qui suffit à attraper une erreur de division par grad_accum.
    """
    a = _copier(config_entrainement, tmp_path / "a")
    a.data.batch_size, a.train.grad_accum, a.train.max_steps = 8, 1, 3
    b = _copier(config_entrainement, tmp_path / "b")
    b.data.batch_size, b.train.grad_accum, b.train.max_steps = 2, 4, 3

    ra, rb = entrainer(a, verbeux=False), entrainer(b, verbeux=False)
    assert abs(ra.perte_finale - rb.perte_finale) < 1.0, (
        f"L'accumulation de gradient ne normalise pas correctement : "
        f"{ra.perte_finale:.3f} contre {rb.perte_finale:.3f}"
    )


def test_divergence_est_detectee(config_entrainement):
    """Une perte non finie doit arrêter l'entraînement avec un message utile,
    plutôt que de brûler des heures de GPU sur des NaN."""
    config_entrainement.train.lr = 1e9
    config_entrainement.train.warmup_steps = 0
    config_entrainement.train.grad_clip = 0.0
    config_entrainement.train.max_steps = 50
    with pytest.raises(RuntimeError, match="divergé"):
        entrainer(config_entrainement, verbeux=False)


# --------------------------------------------------------------------------- #
# Planning du taux d'apprentissage
# --------------------------------------------------------------------------- #


def _cfg_lr(**kwargs) -> FutoConfig:
    return FutoConfig(train=TrainConfig(**kwargs))


def test_echauffement_monte_lineairement():
    cfg = _cfg_lr(lr=1e-3, warmup_steps=100, max_steps=1000, schedule="constant")
    assert taux_apprentissage(0, cfg) == pytest.approx(1e-5)
    assert taux_apprentissage(49, cfg) == pytest.approx(5e-4)
    assert taux_apprentissage(99, cfg) == pytest.approx(1e-3)
    # Jamais nul au premier pas : un taux nul gèlerait le tout premier gradient.
    assert taux_apprentissage(0, cfg) > 0


def test_cosinus_descend_jusquau_minimum():
    cfg = _cfg_lr(lr=1e-3, lr_min_ratio=0.1, warmup_steps=10, max_steps=110,
                  schedule="cosine")
    assert taux_apprentissage(10, cfg) == pytest.approx(1e-3, rel=1e-3)
    assert taux_apprentissage(60, cfg) == pytest.approx(5.5e-4, rel=1e-2)  # mi-parcours
    assert taux_apprentissage(109, cfg) == pytest.approx(1e-4, rel=5e-2)
    # Décroissance monotone après l'échauffement.
    valeurs = [taux_apprentissage(p, cfg) for p in range(10, 110)]
    assert all(a >= b for a, b in zip(valeurs, valeurs[1:], strict=False))


def test_wsd_a_bien_un_plateau():
    cfg = _cfg_lr(lr=1e-3, lr_min_ratio=0.1, warmup_steps=10, max_steps=110,
                  schedule="wsd", decay_steps=20)
    # Plateau entre l'échauffement et le début de la descente.
    assert taux_apprentissage(30, cfg) == pytest.approx(1e-3)
    assert taux_apprentissage(80, cfg) == pytest.approx(1e-3)
    # Puis descente linéaire.
    assert taux_apprentissage(100, cfg) < 1e-3
    assert taux_apprentissage(109, cfg) == pytest.approx(1.45e-4, rel=1e-1)


def test_schedule_inconnu_est_refuse():
    with pytest.raises(ValueError, match="schedule inconnu"):
        TrainConfig(schedule="exponentiel")


def test_echauffement_plus_long_que_lentrainement_est_refuse():
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainConfig(warmup_steps=100, max_steps=50)


# --------------------------------------------------------------------------- #
# Évaluation intercalaire
# --------------------------------------------------------------------------- #


def test_evaluation_est_reproductible(shards, config_modele, tokenizer):
    """Deux évaluations du même modèle doivent donner exactement le même chiffre :
    sinon la courbe de validation oscille sans que le modèle ait changé."""
    from contextlib import nullcontext

    from futo.train import evaluer

    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=32, n_layer=2,
                      n_head=2, n_kv_head=1, d_model=32)
    modele = Futo(cfg)
    chargeur = ChargeurTokens(shards, "val", 32, 2, seed=3)
    a = evaluer(modele, chargeur, 5, torch.device("cpu"), nullcontext())
    b = evaluer(modele, chargeur, 5, torch.device("cpu"), nullcontext())
    assert a == b


def _copier(cfg: FutoConfig, dossier) -> FutoConfig:
    """Duplique une configuration en changeant seulement le dossier de sortie."""
    copie = FutoConfig.from_dict(cfg.to_dict())
    copie.train.dossier_sortie = str(dossier)
    return copie
