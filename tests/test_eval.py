"""Tests de l'évaluation.

Le jeu de paires minimales françaises est vérifié comme du code : c'est un
livrable du dépôt, une faute de français dedans invaliderait la mesure.
"""

from __future__ import annotations

import math
import unicodedata

import pytest
import torch

from futo.data import ChargeurTokens
from futo.eval import (
    ResultatPerplexite,
    charger_sondes,
    mesurer_bits_par_octet,
    mesurer_perplexite,
    mesurer_sondes,
)
from futo.model import Futo

from .conftest import CHEMIN_SONDES

# --------------------------------------------------------------------------- #
# Formules
# --------------------------------------------------------------------------- #


def test_perplexite_est_lexponentielle_de_la_perte():
    r = ResultatPerplexite(perte=math.log(50), n_tokens=100)
    assert r.perplexite == pytest.approx(50.0)


def test_bits_par_token():
    """Une perte de ln(2) nats vaut exactement 1 bit."""
    r = ResultatPerplexite(perte=math.log(2), n_tokens=10)
    assert r.bits_par_token == pytest.approx(1.0)


def test_bits_par_octet_est_independant_du_tokenizer():
    """La même quantité d'information, découpée différemment, donne le même
    nombre de bits par octet. C'est toute la raison d'être de cette métrique.

    Un modèle A : 100 tokens à 2 bits chacun sur 400 octets.
    Un modèle B : 200 tokens à 1 bit chacun sur les mêmes 400 octets.
    Les deux ont dépensé 200 bits pour 400 octets, soit 0,5 bit/octet.
    """
    a = ResultatPerplexite(perte=2 * math.log(2), n_tokens=100, n_octets=400)
    b = ResultatPerplexite(perte=1 * math.log(2), n_tokens=200, n_octets=400)
    assert a.bits_par_octet == pytest.approx(0.5)
    assert b.bits_par_octet == pytest.approx(0.5)
    # Alors que la perplexité par token, elle, diffère du tout au tout.
    assert a.perplexite != pytest.approx(b.perplexite)


def test_bits_par_octet_absent_sans_octets():
    assert ResultatPerplexite(perte=1.0, n_tokens=10).bits_par_octet is None


def test_perplexite_plafonnee_ne_deborde_pas():
    """Une perte aberrante ne doit pas produire un infini qui casse l'affichage."""
    assert math.isfinite(ResultatPerplexite(perte=1e6, n_tokens=1).perplexite)


# --------------------------------------------------------------------------- #
# Mesures sur un vrai modèle
# --------------------------------------------------------------------------- #


def test_perplexite_dun_modele_neuf_vaut_le_vocabulaire(shards, tokenizer):
    """Un modèle non entraîné a une perplexité proche de la taille du vocabulaire :
    il tire au hasard uniformément."""
    torch.manual_seed(0)
    from futo.config import ModelConfig

    modele = Futo(ModelConfig(vocab_size=tokenizer.vocab_size, block_size=32,
                              n_layer=2, n_head=2, n_kv_head=1, d_model=32))
    chargeur = ChargeurTokens(shards, "val", 32, 2, seed=3)
    r = mesurer_perplexite(modele, chargeur, n_lots=4)
    assert r.n_tokens == 4 * 2 * 32
    # Large tolérance : les poids liés donnent un léger biais dès l'init.
    assert 0.3 * tokenizer.vocab_size < r.perplexite < 3 * tokenizer.vocab_size


def test_bits_par_octet_sur_texte_brut(tokenizer, texte_court):
    torch.manual_seed(0)
    from futo.config import ModelConfig

    modele = Futo(ModelConfig(vocab_size=tokenizer.vocab_size, block_size=64,
                              n_layer=2, n_head=2, n_kv_head=1, d_model=32))
    r = mesurer_bits_par_octet(modele, tokenizer, [texte_court] * 3)
    assert r.n_octets == 3 * len(unicodedata.normalize("NFC", texte_court).encode("utf-8"))
    assert r.bits_par_octet is not None and r.bits_par_octet > 0
    # Un modèle neuf ne peut pas faire mieux que le hasard : plusieurs bits par
    # octet. Un bon modèle français descend nettement en dessous de 1.
    assert r.bits_par_octet > 1.0


def test_bits_par_octet_refuse_du_vide(tokenizer, modele):
    with pytest.raises(ValueError, match="Aucun token"):
        mesurer_bits_par_octet(modele, tokenizer, ["", ""])


# --------------------------------------------------------------------------- #
# Le jeu de sondes françaises
# --------------------------------------------------------------------------- #


def test_le_fichier_de_sondes_est_livre():
    assert CHEMIN_SONDES.exists(), "Le jeu de paires minimales doit être versionné."


def test_sondes_bien_formees():
    paires = charger_sondes(CHEMIN_SONDES)
    assert len(paires) >= 50, f"Seulement {len(paires)} paires : trop peu pour mesurer."
    for p in paires:
        assert p.correct != p.incorrect
        assert p.correct.strip() and p.incorrect.strip()
        assert p.note, f"Paire sans explication : {p.correct}"
        # Une paire doit être minimale : les deux phrases doivent se ressembler.
        assert abs(len(p.correct) - len(p.incorrect)) < 40, (
            f"Paire déséquilibrée en longueur : {p.correct!r} / {p.incorrect!r}"
        )


def test_sondes_couvrent_les_difficultes_du_francais():
    """Le jeu doit couvrir les phénomènes qui font le français, pas seulement
    l'accord le plus facile."""
    paires = charger_sondes(CHEMIN_SONDES)
    phenomenes = {p.phenomene for p in paires}
    attendus = {
        "accord-sujet-verbe-distance",
        "accord-participe-etre",
        "accord-participe-avoir-cod-antepose",
        "genre-determinant",
        "subjonctif-apres-conjonction",
        "negation",
        "ordre-clitiques",
        "elision",
        "conjugaison-temps",
        "pronom-relatif",
    }
    manquants = attendus - phenomenes
    assert not manquants, f"Phénomènes non couverts : {sorted(manquants)}"
    # Chaque phénomène doit avoir plusieurs paires : une seule ne mesure rien.
    from collections import Counter

    compte = Counter(p.phenomene for p in paires)
    trop_rares = [ph for ph, n in compte.items() if n < 2]
    assert not trop_rares, f"Phénomènes à une seule paire : {trop_rares}"


def test_sondes_contiennent_les_pieges_symetriques():
    """Le jeu doit aussi contenir les cas INVERSES, sinon un modèle qui accorde
    tout systématiquement obtiendrait un score parfait sans rien comprendre."""
    paires = charger_sondes(CHEMIN_SONDES)
    phenomenes = {p.phenomene for p in paires}
    assert "accord-participe-avoir-cod-postpose" in phenomenes, (
        "Sans le cas « COD placé après » (pas d'accord), le test récompense "
        "l'accord systématique."
    )
    assert "indicatif-attendu" in phenomenes, (
        "Sans les cas où l'indicatif est correct, le test récompense le "
        "subjonctif systématique."
    )
    assert "elision-interdite" in phenomenes


def test_sondes_sont_encodables_sans_perte(tokenizer):
    """Toutes les phrases doivent traverser le tokenizer sans altération."""
    for p in charger_sondes(CHEMIN_SONDES):
        for phrase in (p.correct, p.incorrect):
            attendu = unicodedata.normalize("NFC", phrase)
            assert tokenizer.decoder(tokenizer.encoder(attendu)) == attendu


def test_mesure_des_sondes_fonctionne(tokenizer):
    """Un modèle neuf doit obtenir un score voisin du hasard, sans planter."""
    torch.manual_seed(0)
    from futo.config import ModelConfig

    modele = Futo(ModelConfig(vocab_size=tokenizer.vocab_size, block_size=128,
                              n_layer=2, n_head=2, n_kv_head=1, d_model=32))
    r = mesurer_sondes(modele, tokenizer, CHEMIN_SONDES)

    assert r.n_paires >= 50
    assert 0.0 <= r.justesse <= 1.0
    assert 0.0 <= r.justesse_par_octet <= 1.0
    assert r.par_phenomene
    assert sum(total for _, total in r.par_phenomene.values()) == r.n_paires
    assert "hasard" in str(r)


def test_sondes_detectent_une_preference(tokenizer):
    """Le mécanisme doit vraiment comparer : un modèle qui préfère nettement une
    phrase doit être détecté comme tel.

    On fabrique la préférence en entraînant un modèle jouet à mémoriser les
    phrases correctes uniquement. S'il ne les préfère pas ensuite, c'est que la
    comparaison de log-vraisemblances est cassée.
    """
    torch.manual_seed(0)
    from futo.config import ModelConfig

    paires = charger_sondes(CHEMIN_SONDES)[:12]
    modele = Futo(ModelConfig(vocab_size=tokenizer.vocab_size, block_size=128,
                              n_layer=2, n_head=2, n_kv_head=2, d_model=64))
    modele.train()
    optimiseur = torch.optim.AdamW(modele.groupes_parametres(0.0), lr=3e-3)

    lots = []
    for p in paires:
        ids = [tokenizer.id_fin] + tokenizer.encoder(unicodedata.normalize("NFC", p.correct))
        ids = ids[:129]
        lots.append(torch.tensor([ids], dtype=torch.long))

    for _ in range(60):
        for lot in lots:
            optimiseur.zero_grad(set_to_none=True)
            _, perte = modele(lot[:, :-1], cibles=lot[:, 1:])
            perte.backward()
            optimiseur.step()

    r = mesurer_sondes(modele, tokenizer, paires)
    assert r.justesse > 0.8, (
        f"Le modèle a mémorisé les phrases correctes mais n'obtient que "
        f"{r.justesse * 100:.0f} % : la comparaison est probablement cassée."
    )
    assert r.marge_moyenne > 0


def test_sondes_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        charger_sondes(tmp_path / "absent.jsonl")


def test_sondes_ligne_invalide_est_signalee(tmp_path):
    chemin = tmp_path / "s.jsonl"
    chemin.write_text('{"phenomene": "x", "correct": "a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="champs manquants"):
        charger_sondes(chemin)


def test_sondes_phrases_identiques_sont_refusees(tmp_path):
    chemin = tmp_path / "s.jsonl"
    chemin.write_text(
        '{"phenomene": "x", "correct": "a", "incorrect": "a"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identiques"):
        charger_sondes(chemin)


def test_les_suggestions_comptent_le_rang_pas_la_probabilite(tmp_path):
    """Un clavier ne montre que k cases : seul le RANG du bon mot compte.

    Vérifié sur un modèle truqué dont on connaît l'ordre des logits : la cible
    placée au rang 3 doit compter pour top-3, top-4 et top-5, jamais pour top-1.
    """
    import torch

    from futo.eval import mesurer_suggestions

    class ModeleTruque(torch.nn.Module):
        """Renvoie toujours le même classement : 0 > 1 > 2 > 3 > …"""

        def __init__(self, vocab):
            super().__init__()
            self.vocab = vocab
            self.bidon = torch.nn.Parameter(torch.zeros(1))

        def forward(self, entree, cibles=None, tous_les_pas=False):
            b, t = entree.shape
            rangs = torch.arange(self.vocab, dtype=torch.float32)
            logits = (-rangs).expand(b, t, self.vocab).contiguous()
            return logits, None

    class TokenizerTruque:
        vocab_size = 8

        def decoder(self, ids, sauter_speciaux=True):
            # seuls les identifiants pairs ouvrent un mot
            return " mot" if ids[0] % 2 == 0 else "suite"

    class ChargeurTruque:
        def lot(self, i):
            entree = torch.zeros(1, 4, dtype=torch.long)
            # cibles : rang 0, rang 2, rang 3, rang 5 dans le classement
            cible = torch.tensor([[0, 2, 3, 5]], dtype=torch.long)
            return entree, cible

    r = mesurer_suggestions(ModeleTruque(8), ChargeurTruque(), TokenizerTruque(), 1,
                            ks=(1, 3, 4, 5), peripherique=torch.device("cpu"))
    assert r.n_tokens == 4
    assert r.reussites[1] == 1           # seule la cible 0 est en tête
    assert r.reussites[3] == 2           # cibles 0 et 2
    assert r.reussites[4] == 3           # + cible 3
    assert r.reussites[5] == 3           # cible 5 est au rang 6, toujours dehors
    # débuts de mot : identifiants pairs, soit les cibles 0 et 2
    assert r.n_debuts_de_mot == 2
    assert r.reussites_mots[4] == 2
    assert 0.0 < r.taux(4) < 1.0
    assert r.taux_mots(1) == 0.5
