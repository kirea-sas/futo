"""Tests du transformeur.

Les trois premiers sont les plus importants du dépôt. Un modèle de langue peut
être profondément faux tout en produisant des tenseurs de la bonne forme et une
perte qui descend : ces tests-là attrapent les fautes qui, sinon, ne se voient
qu'après plusieurs jours de GPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from futo.config import ModelConfig
from futo.model import CacheKV, Futo, appliquer_rope, construire_rope


# --------------------------------------------------------------------------- #
# Correction fondamentale
# --------------------------------------------------------------------------- #


def test_causalite_aucune_fuite_du_futur(modele):
    """Modifier le token en position t ne doit RIEN changer aux logits avant t.

    C'est la propriété qui définit un modèle causal. Si elle est fausse, le
    modèle triche pendant l'entraînement — la perte s'effondre magnifiquement —
    et il devient inutilisable en génération, où le futur n'existe pas.
    """
    torch.manual_seed(0)
    T = 32
    entree = torch.randint(0, modele.cfg.vocab_size, (1, T))

    logits_a, _ = modele(entree, cibles=entree)

    coupure = 20
    modifie = entree.clone()
    # On change tout à partir de la coupure, en s'assurant que ça change vraiment.
    modifie[:, coupure:] = (modifie[:, coupure:] + 1) % modele.cfg.vocab_size
    logits_b, _ = modele(modifie, cibles=modifie)

    avant = torch.allclose(logits_a[:, :coupure], logits_b[:, :coupure], atol=1e-5)
    assert avant, "Fuite d'information : les logits avant la coupure ont changé."

    apres_differe = not torch.allclose(
        logits_a[:, coupure:], logits_b[:, coupure:], atol=1e-5
    )
    assert apres_differe, (
        "Les logits après la coupure sont identiques : le test ne prouve rien, "
        "l'entrée n'a probablement pas été modifiée."
    )


def test_cache_kv_equivaut_a_la_passe_complete(modele):
    """Générer token par token avec le cache doit donner les mêmes logits
    qu'une seule passe sur la séquence entière.

    C'est le test qui attrape le piège n° 2 : `is_causal=True` employé avec un
    cache KV, où le masque mal aligné produit des logits silencieusement faux.
    Le modèle s'entraîne alors correctement mais génère n'importe quoi.
    """
    torch.manual_seed(0)
    T = 16
    entree = torch.randint(0, modele.cfg.vocab_size, (2, T))

    logits_complets, _ = modele(entree, cibles=entree)

    # Décodage incrémental : un token à la fois, en repartant de zéro.
    cache = CacheKV.vide(modele.cfg.n_layer)
    logits_incrementaux = []
    for t in range(T):
        logits, _ = modele(entree[:, t : t + 1], cache=cache)
        logits_incrementaux.append(logits[:, -1, :])
    logits_incrementaux = torch.stack(logits_incrementaux, dim=1)

    ecart = (logits_complets - logits_incrementaux).abs().max().item()
    assert ecart < 1e-4, (
        f"Le cache KV diverge de la passe complète (écart max {ecart:.2e}). "
        f"Le masque causal est probablement mal aligné en décodage."
    )


def test_cache_kv_preremplissage_par_morceaux(modele):
    """Le cas intermédiaire : 1 < q_len < kv_len.

    Préremplir en deux morceaux doit donner le même résultat qu'en un seul. Ce
    cas n'est couvert ni par `is_causal=True` ni par « pas de masque » : il faut
    le masque explicitement décalé.
    """
    torch.manual_seed(0)
    T = 12
    entree = torch.randint(0, modele.cfg.vocab_size, (1, T))
    logits_complets, _ = modele(entree, cibles=entree)

    cache = CacheKV.vide(modele.cfg.n_layer)
    modele(entree[:, :7], cache=cache)  # premier morceau
    # Second morceau : q_len = 5, kv_len = 12 — le cas du masque décalé.
    logits, _ = modele(entree[:, 7:], cache=cache, tous_les_pas=True)

    assert logits.shape == logits_complets[:, 7:].shape
    ecart = (logits_complets[:, 7:] - logits).abs().max().item()
    assert ecart < 1e-4, (
        f"Le préremplissage par morceaux diverge (écart max {ecart:.2e})."
    )


def test_rope_ne_depend_que_de_la_distance(config_modele):
    """RoPE doit rendre le produit scalaire q·k invariant par translation.

    C'est toute la raison d'être des positions rotatives : le score entre les
    positions m et n ne doit dépendre que de m − n. Si l'implémentation est
    fausse (mauvais appariement des dimensions, angles en bf16), la propriété
    tombe et le modèle perd sa capacité d'extrapolation.
    """
    torch.manual_seed(0)
    head_dim = 16
    cos, sin = construire_rope(head_dim, 64, theta=10_000.0)

    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def score(m: int, n: int) -> float:
        qm = appliquer_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = appliquer_rope(k, cos[n : n + 1], sin[n : n + 1])
        return float((qm * kn).sum())

    # Trois paires de même écart (5), à des positions très différentes.
    references = [score(5, 0), score(20, 15), score(50, 45)]
    for valeur in references[1:]:
        assert abs(valeur - references[0]) < 1e-4, (
            f"Le score dépend de la position absolue et pas seulement de l'écart : "
            f"{references}"
        )

    # Et un écart différent doit, lui, donner un score différent.
    assert abs(score(10, 0) - references[0]) > 1e-6


# --------------------------------------------------------------------------- #
# Formes, comptages, initialisation
# --------------------------------------------------------------------------- #


def test_formes_de_bout_en_bout(modele):
    entree = torch.randint(0, modele.cfg.vocab_size, (3, 17))
    logits, perte = modele(entree, cibles=entree)
    assert logits.shape == (3, 17, modele.cfg.vocab_size)
    assert perte.ndim == 0 and torch.isfinite(perte)

    # Sans cibles, seul le dernier pas est calculé : c'est une optimisation
    # importante en génération, et une régression facile à introduire.
    logits_gen, perte_gen = modele(entree)
    assert logits_gen.shape == (3, 1, modele.cfg.vocab_size)
    assert perte_gen is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_layer": 2, "n_head": 4, "n_kv_head": 2, "d_model": 64, "vocab_size": 500},
        {"n_layer": 3, "n_head": 8, "n_kv_head": 8, "d_model": 128, "vocab_size": 1000},
        {"n_layer": 1, "n_head": 6, "n_kv_head": 1, "d_model": 96, "vocab_size": 777},
    ],
)
def test_comptage_parametres_analytique(kwargs):
    """La formule de `ModelConfig.nombre_parametres` doit coller au réel.

    On s'en sert pour dimensionner un entraînement avant de louer un GPU : une
    formule fausse se paie en euros.
    """
    for tie in (True, False):
        cfg = ModelConfig(block_size=32, tie_weights=tie, **kwargs)
        reel = Futo(cfg).nombre_parametres()
        assert cfg.nombre_parametres() == reel, (
            f"Formule fausse (tie_weights={tie}) : "
            f"{cfg.nombre_parametres()} annoncés, {reel} réels."
        )


def test_poids_lies_partagent_bien_le_meme_tenseur():
    cfg = ModelConfig(vocab_size=100, block_size=16, n_layer=1, n_head=2,
                      n_kv_head=1, d_model=32, tie_weights=True)
    m = Futo(cfg)
    assert m.tete.weight is m.embeddings.weight

    cfg_delie = ModelConfig(vocab_size=100, block_size=16, n_layer=1, n_head=2,
                            n_kv_head=1, d_model=32, tie_weights=False)
    m_delie = Futo(cfg_delie)
    assert m_delie.tete.weight is not m_delie.embeddings.weight
    # Délier ajoute exactement V×d paramètres.
    assert m_delie.nombre_parametres() - m.nombre_parametres() == 100 * 32


def test_perte_initiale_proche_du_hasard():
    """À l'initialisation, le modèle ne sait rien : la perte doit valoir ln(V).

    Un écart important signale une initialisation cassée — échelle trop grande,
    logits saturés — qui rend les premiers pas instables.

    La cible est tirée indépendamment de l'entrée : si on prenait cibles=entrée,
    les poids liés donneraient au modèle un avantage trivial (le produit
    scalaire d'un embedding avec lui-même est maximal) et la perte serait
    artificiellement basse.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=1024, block_size=32, n_layer=2, n_head=4,
                      n_kv_head=2, d_model=64)
    m = Futo(cfg)
    m.eval()
    entree = torch.randint(0, cfg.vocab_size, (8, 32))
    cible = torch.randint(0, cfg.vocab_size, (8, 32))
    _, perte = m(entree, cibles=cible)

    attendu = math.log(cfg.vocab_size)
    assert abs(perte.item() - attendu) < 0.5, (
        f"Perte initiale {perte.item():.3f}, attendue ≈ {attendu:.3f} = ln(V)."
    )


def test_groupes_de_parametres_excluent_les_normes(modele):
    """Le weight decay ne doit toucher que les tenseurs de dimension ≥ 2."""
    groupes = modele.groupes_parametres(weight_decay=0.1)
    assert len(groupes) == 2
    avec, sans = groupes
    assert avec["weight_decay"] == 0.1
    assert sans["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in avec["params"])
    assert all(p.dim() < 2 for p in sans["params"])
    # Rien ne doit être oublié, ni compté deux fois.
    total = sum(p.numel() for p in avec["params"]) + sum(p.numel() for p in sans["params"])
    assert total == modele.nombre_parametres()


def test_gqa_reduit_bien_les_parametres():
    """Passer de MHA à GQA doit réduire les projections K et V, pas Q ni O."""
    commun = dict(vocab_size=100, block_size=16, n_layer=1, d_model=64, n_head=8)
    mha = ModelConfig(n_kv_head=8, **commun)
    gqa = ModelConfig(n_kv_head=2, **commun)
    # Économie attendue : 2 × d_model × (8 − 2) × head_dim.
    attendu = 2 * 64 * (8 - 2) * 8
    assert mha.nombre_parametres() - gqa.nombre_parametres() == attendu


# --------------------------------------------------------------------------- #
# Garde-fous
# --------------------------------------------------------------------------- #


def test_configuration_incoherente_est_refusee():
    with pytest.raises(ValueError, match="multiple de n_kv_head"):
        ModelConfig(n_head=6, n_kv_head=4)
    with pytest.raises(ValueError, match="divisible par n_head"):
        ModelConfig(d_model=100, n_head=8, n_kv_head=8)


def test_sequence_trop_longue_leve_une_erreur_claire(modele):
    trop_long = torch.randint(0, modele.cfg.vocab_size, (1, modele.cfg.block_size + 1))
    with pytest.raises(ValueError, match="Séquence trop longue"):
        modele(trop_long, cibles=trop_long)


def test_dff_calcule_est_un_multiple():
    cfg = ModelConfig(d_model=768, d_ff=None, d_ff_multiple=64)
    assert cfg.d_ff % 64 == 0
    assert cfg.d_ff >= 8 * 768 / 3  # arrondi vers le haut, jamais vers le bas


# --------------------------------------------------------------------------- #
# Génération
# --------------------------------------------------------------------------- #


def test_generation_deterministe_avec_graine(modele):
    amorce = torch.randint(0, modele.cfg.vocab_size, (1, 4))
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = modele.generer(amorce, max_tokens=10, temperature=0.9, generateur=g1)
    b = modele.generer(amorce, max_tokens=10, temperature=0.9, generateur=g2)
    assert torch.equal(a, b)


def test_generation_gloutonne_est_reproductible(modele):
    """À température nulle, on prend l'argmax : aucun aléa ne doit subsister."""
    amorce = torch.randint(0, modele.cfg.vocab_size, (1, 4))
    a = modele.generer(amorce, max_tokens=8, temperature=0.0)
    b = modele.generer(amorce, max_tokens=8, temperature=0.0)
    assert torch.equal(a, b)
    assert a.shape == (1, 12)


def test_generation_conserve_lamorce(modele):
    amorce = torch.randint(0, modele.cfg.vocab_size, (2, 5))
    sortie = modele.generer(amorce, max_tokens=6, temperature=0.7)
    assert torch.equal(sortie[:, :5], amorce)
    assert sortie.shape == (2, 11)


def test_generation_sarrete_au_token_de_fin(modele):
    """Avec un token de fin, la génération s'arrête et remplit avec ce token."""
    amorce = torch.randint(0, modele.cfg.vocab_size, (1, 3))
    # Température très basse et un token de fin très probable : on force le cas.
    with torch.no_grad():
        modele.tete.weight[0] += 50.0  # rend le token 0 quasi certain
    sortie = modele.generer(amorce, max_tokens=10, temperature=0.1, top_k=1, token_fin=0)
    assert sortie.shape[1] < 13, "La génération aurait dû s'arrêter avant la limite."


def test_generation_ne_depasse_pas_le_contexte(modele):
    amorce = torch.randint(0, modele.cfg.vocab_size, (1, modele.cfg.block_size - 3))
    sortie = modele.generer(amorce, max_tokens=100, temperature=0.8)
    assert sortie.shape[1] <= modele.cfg.block_size


def test_redimensionner_le_contexte(modele):
    modele.redimensionner_contexte(128)
    assert modele.cfg.block_size == 128
    assert modele.rope_cos.shape[0] == 128
    entree = torch.randint(0, modele.cfg.vocab_size, (1, 100))
    logits, _ = modele(entree, cibles=entree)
    assert logits.shape[1] == 100


def test_mode_entrainement_restaure_apres_generation(modele):
    modele.train()
    modele.generer(torch.zeros(1, 2, dtype=torch.long), max_tokens=2)
    assert modele.training, "generer() doit rendre le modèle dans l'état où il l'a trouvé."
