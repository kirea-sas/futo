"""Tests du tokenizer français.

L'exigence numéro un est qu'il ne perde jamais rien : tout ce qui entre doit
ressortir à l'identique. La deuxième est qu'il traite correctement les
particularités du français, ce que ne fait aucun tokenizer anglophone.
"""

from __future__ import annotations

import unicodedata

import pytest

from futo.tokenizer import (
    MOTIF_FRANCAIS,
    TOKENS_SPECIAUX,
    mesurer_fertilite,
)

# Les difficultés du français, une par cas, avec ce qu'elles éprouvent.
CAS_FRANCAIS = [
    ("L'homme qu'il a vu aujourd'hui.", "élisions avec apostrophe droite"),
    ("L’homme qu’il a vu aujourd’hui.", "élisions avec apostrophe typographique"),
    ("Le cœur de l'œuvre de ma sœur.", "ligature œ"),
    ("Ex æquo, un nævus, et cætera.", "ligature æ"),
    ("À Nîmes, Ève a hâté son départ.", "majuscules accentuées"),
    ("Où qu'il aille, il coûte 12 %.", "accents rares et pourcentage"),
    ("« Bonjour ! » dit-il ; puis : « Ça va ? »", "guillemets français et espaces fines"),
    ("Il en reste 1 234,56 € sur 10 000.", "nombres à la française"),
    ("Y a-t-il quelqu'un ? Vas-y, va-t'en !", "t euphonique et clitiques"),
    ("Donne-le-moi, celui-ci, arc-en-ciel.", "chaînes de traits d'union"),
    ("C'est-à-dire presqu'île et jusqu'au-boutiste.", "élisions longues"),
    ("Naïve, aiguë, ambiguïté, Noël.", "trémas"),
    ("SNCF, INSEE, M. Dupont, n° 12, Dr Ré.", "sigles et abréviations"),
    ("def f(x):\n    return x ** 2  # café", "code mêlé de français"),
    ("Emoji 🇫🇷 et 😀 dans du texte.", "hors du plan multilingue de base"),
    ("Ligne un\nLigne deux\n\nParagraphe.", "sauts de ligne"),
    ("   espaces   multiples   ", "espaces répétés et en bordure"),
    ("", "chaîne vide"),
]


# --------------------------------------------------------------------------- #
# Aller-retour : la propriété non négociable
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("texte,description", CAS_FRANCAIS, ids=[c[1] for c in CAS_FRANCAIS])
def test_aller_retour_exact(tokenizer, texte, description):
    """encoder puis décoder doit rendre le texte d'origine, octet pour octet.

    Le BPE au niveau octet le garantit : le vocabulaire de départ contient les
    256 octets, donc rien ne peut être hors vocabulaire. La seule transformation
    admise est la normalisation NFC, appliquée en amont ici pour que la
    comparaison soit juste.
    """
    attendu = unicodedata.normalize("NFC", texte)
    obtenu = tokenizer.decoder(tokenizer.encoder(attendu))
    assert obtenu == attendu, f"Aller-retour cassé sur : {description}"


def test_aller_retour_sur_tout_le_corpus(tokenizer, texte_francais):
    """La même garantie, sur l'intégralité du corpus d'exemple."""
    attendu = unicodedata.normalize("NFC", texte_francais)
    assert tokenizer.decoder(tokenizer.encoder(attendu)) == attendu


def test_aucun_octet_nest_hors_vocabulaire(tokenizer):
    """Les 256 octets doivent être encodables, même absents du corpus."""
    texte = "".join(chr(c) for c in range(32, 127)) + "þÿøæ𝄞"
    texte = unicodedata.normalize("NFC", texte)
    assert tokenizer.decoder(tokenizer.encoder(texte)) == texte


def test_les_deux_apostrophes_sont_distinguees(tokenizer):
    """Droite et typographique ne doivent pas être confondues.

    Les normaliser l'une vers l'autre simplifierait le vocabulaire, mais rendrait
    la restitution du texte impossible : c'est un choix assumé, on le vérifie.
    """
    droite = "L'homme n'est pas d'accord."
    typo = "L’homme n’est pas d’accord."
    assert tokenizer.decoder(tokenizer.encoder(droite)) == droite
    assert tokenizer.decoder(tokenizer.encoder(typo)) == typo
    assert tokenizer.encoder(droite) != tokenizer.encoder(typo)


# --------------------------------------------------------------------------- #
# La découpe française
# --------------------------------------------------------------------------- #


def _decouper(texte: str) -> list[str]:
    from tokenizers import Regex, pre_tokenizers

    decoupeur = pre_tokenizers.Split(
        pattern=Regex(MOTIF_FRANCAIS), behavior="isolated", invert=False
    )
    return [morceau for morceau, _ in decoupeur.pre_tokenize_str(texte)]


@pytest.mark.parametrize(
    "texte,attendu",
    [
        ("L'homme", ["L'", "homme"]),
        ("L’homme", ["L’", "homme"]),
        ("qu'il", ["qu'", "il"]),
        ("aujourd'hui", ["aujourd'", "hui"]),
        ("jusqu'à", ["jusqu'", "à"]),
        ("lorsqu'on", ["lorsqu'", "on"]),
        ("presqu'île", ["presqu'", "île"]),
        ("d'accord", ["d'", "accord"]),
        ("n'est", ["n'", "est"]),
        ("s'il", ["s'", "il"]),
    ],
)
def test_elision_reste_soudee_au_proclitique(texte, attendu):
    """« l'homme » doit donner « l' » + « homme », jamais « l » + « 'homme ».

    C'est la différence concrète avec un tokenizer anglophone, dont la règle de
    contraction (`'s`, `'t`, `'ll`) attache l'apostrophe au morceau de DROITE.
    """
    assert _decouper(texte) == attendu


@pytest.mark.parametrize(
    "texte,attendu",
    [
        ("dit-il", ["dit", "-il"]),
        ("vas-y", ["vas", "-y"]),
        ("a-t-il", ["a", "-t-il"]),
        ("viendra-t-elle", ["viendra", "-t-elle"]),
        ("va-t'en", ["va", "-t'en"]),
        ("celui-ci", ["celui", "-ci"]),
        ("donne-le", ["donne", "-le"]),
    ],
)
def test_clitiques_a_trait_dunion(texte, attendu):
    """Les pronoms accrochés par un trait d'union restent d'un bloc."""
    assert _decouper(texte) == attendu


def test_chiffres_coupes_par_groupes_de_trois():
    """Un nombre ne doit jamais former un seul token, quelle que soit sa longueur.

    Sans cette règle, le modèle apprend par cœur les nombres fréquents et calcule
    d'autant plus mal.
    """
    for morceau in _decouper("123456789"):
        assert len(morceau.strip()) <= 3, f"Groupe de chiffres trop long : {morceau!r}"
    assert _decouper("1 234,56") == ["1", " 234", ",", "56"]


def test_le_motif_couvre_tout_le_texte(texte_court):
    """Aucun caractère ne doit être perdu par la découpe préalable."""
    assert "".join(_decouper(texte_court)) == texte_court


# --------------------------------------------------------------------------- #
# Tokens spéciaux, vocabulaire, mesures
# --------------------------------------------------------------------------- #


def test_identifiants_speciaux_stables(tokenizer):
    """L'ordre des tokens spéciaux fixe leurs identifiants : il ne doit pas bouger.

    Changer cet ordre rendrait tous les checkpoints existants incompatibles avec
    le tokenizer, sans qu'aucune erreur ne se déclenche.
    """
    assert TOKENS_SPECIAUX[0] == "<|fin_de_texte|>"
    assert TOKENS_SPECIAUX[1] == "<|rembourrage|>"
    assert TOKENS_SPECIAUX[2] == "<|debut_de_texte|>"
    assert tokenizer.id_fin == 0
    assert tokenizer.id_rembourrage == 1
    assert tokenizer.id_debut == 2
    # Des emplacements sont réservés pour la suite (dialogue, rôles, outils).
    assert sum(1 for t in TOKENS_SPECIAUX if t.startswith("<|reserve_")) >= 8


def test_ajout_du_token_de_fin(tokenizer):
    sans = tokenizer.encoder("Bonjour.")
    avec = tokenizer.encoder("Bonjour.", ajouter_fin=True)
    assert avec == sans + [tokenizer.id_fin]


def test_encodage_par_lot_identique_a_lunitaire(tokenizer):
    textes = ["L'homme.", "Le cœur.", "1 234,56 €"]
    assert tokenizer.encoder_lot(textes) == [tokenizer.encoder(t) for t in textes]


def test_dtype_des_shards_suit_le_vocabulaire(tokenizer):
    import numpy as np

    assert tokenizer.vocab_size <= 65_535
    assert tokenizer.dtype_shards is np.uint16


def test_fertilite_est_plausible(tokenizer, texte_francais):
    """La fertilité doit rester dans une fourchette crédible pour du français.

    Le tokenizer des tests est entraîné sur 150 Ko avec un vocabulaire de 2 048 :
    c'est minuscule, on n'attend donc pas les chiffres d'un vrai modèle. On
    vérifie seulement que la mesure est cohérente et que le calcul ne s'est pas
    trompé d'unité.
    """
    f = mesurer_fertilite(tokenizer, texte_francais[:200_000])
    assert 1.0 <= f.tokens_par_mot <= 8.0, f"Fertilité invraisemblable : {f}"
    assert 1.0 <= f.octets_par_token <= 10.0
    assert 0.0 <= f.mots_en_un_token <= 1.0
    assert f.n_mots > 100


def test_vocabulaire_trop_petit_est_refuse(tmp_path, texte_court):
    """Un vocabulaire inférieur aux 256 octets + spéciaux ne peut pas fonctionner."""
    from futo.tokenizer import entrainer_tokenizer

    source = tmp_path / "c.txt"
    source.write_text(texte_court, encoding="utf-8")
    with pytest.raises(ValueError, match="doit dépasser"):
        entrainer_tokenizer([source], tmp_path / "t.json", vocab_size=100, verbeux=False)


def test_corpus_introuvable_leve_une_erreur_claire(tmp_path):
    from futo.tokenizer import entrainer_tokenizer

    with pytest.raises(FileNotFoundError, match="introuvable"):
        entrainer_tokenizer([tmp_path / "absent.txt"], tmp_path / "t.json", verbeux=False)


def test_fiche_ecrite_a_cote_du_tokenizer(tokenizer):
    """Une fiche JSON doit documenter d'où sort le vocabulaire."""
    import json

    fiche = tokenizer.chemin.with_suffix(".fiche.json")
    assert fiche.exists()
    contenu = json.loads(fiche.read_text(encoding="utf-8"))
    assert contenu["normalisation"] == "NFC"
    assert contenu["motif_decoupe"] == MOTIF_FRANCAIS
    assert contenu["vocab_size"] == tokenizer.vocab_size
