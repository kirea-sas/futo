"""Tests du contrôle de corpus.

Un outil de contrôle qui ne détecte rien est pire qu'aucun outil : il donne
l'assurance sans la vérification. Chaque test vérifie donc qu'une saleté
précise est bien VUE, et qu'un corpus propre est bien déclaré propre.
"""

from __future__ import annotations

import json

import pytest

from futo.controle import (
    TRACES_DE_BALISAGE,
    controler_corpus,
    extraire_echantillon,
)

PROPRE = (
    "L'histoire de Lyon commence bien avant notre ère. La ville, au confluent "
    "du Rhône et de la Saône, a connu de nombreux bouleversements. Son cœur "
    "historique est classé au patrimoine mondial depuis 1998, et les traboules "
    "en sont l'emblème le plus visité. « Une ville de passage », disait-on."
)


def _jsonl(tmp_path, textes: list[str], nom: str = "c.jsonl"):
    chemin = tmp_path / nom
    chemin.write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in textes),
        encoding="utf-8",
    )
    return chemin


# --------------------------------------------------------------------------- #
# Détection des saletés
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "saleté,cle_attendue",
    [
        ("Reste {{un modele}} ici.", "modele_ouvrant"),
        ("Fermeture }} orpheline.", "modele_fermant"),
        ("Un [[lien]] non traité.", "lien_ouvrant"),
        ("Un tableau {| bancal.", "tableau"),
        ("Une <ref>note</ref>.", "balise_ref"),
        ("Du <div>HTML</div>.", "balise_html"),
        ("Une entit&eacute; ? &nbsp; oui.", "entite_html"),
        ("Du '''gras''' resté.", "gras_italique"),
        ("== Un titre ==", "titre_wiki"),
        ("Voir https://exemple.fr/page/longue", "url_nue"),
    ],
)
def test_chaque_salete_est_detectee(tmp_path, saleté, cle_attendue):
    """Le contrôle doit VOIR ce qu'il prétend chercher."""
    # Séparation par un saut de ligne : certains motifs, comme les titres de
    # section, ne se reconnaissent qu'en début de ligne.
    rapport = controler_corpus([_jsonl(tmp_path, [PROPRE + "\n" + saleté])])
    assert rapport.traces.get(cle_attendue, 0) > 0, (
        f"« {saleté} » n'a pas été détecté comme {cle_attendue}."
    )
    assert cle_attendue in rapport.exemples
    assert rapport.docs_touches[cle_attendue] == 1


def test_un_corpus_propre_est_declare_propre(tmp_path):
    rapport = controler_corpus([_jsonl(tmp_path, [PROPRE] * 5)])
    assert not {c: n for c, n in rapport.traces.items() if n}
    ok, message = rapport.verdict()
    assert ok and "propre" in message


def test_un_corpus_polluee_est_refuse(tmp_path):
    """Au-delà de 5 % de documents touchés, le verdict doit être négatif."""
    textes = [PROPRE + " {{modele}}" for _ in range(10)]
    rapport = controler_corpus([_jsonl(tmp_path, textes)])
    ok, message = rapport.verdict()
    assert not ok
    assert "à revoir" in message


def test_un_corpus_sans_accents_est_signale(tmp_path):
    """Un « corpus français » sans accents trahit un problème d'encodage ou de
    langue — mieux vaut le voir avant l'entraînement qu'après."""
    anglais = "The history of the city begins long before our era, at the river."
    rapport = controler_corpus([_jsonl(tmp_path, [anglais] * 5)])
    ok, message = rapport.verdict()
    assert not ok
    assert "français" in message


# --------------------------------------------------------------------------- #
# Comptages
# --------------------------------------------------------------------------- #


def test_comptages_de_base(tmp_path):
    rapport = controler_corpus([_jsonl(tmp_path, [PROPRE, PROPRE + " Encore."])])
    assert rapport.n_documents == 2
    assert rapport.n_octets == len(PROPRE.encode()) + len((PROPRE + " Encore.").encode())
    assert rapport.n_mots > 50
    assert len(rapport.longueurs) == 2
    assert rapport.tokens_estimes == pytest.approx(rapport.n_octets / 3.56)


def test_typographie_francaise_comptee(tmp_path):
    texte = "L'œuvre de la sœur — « Bonjour » — coûte 12 % de plus qu'hier."
    rapport = controler_corpus([_jsonl(tmp_path, [texte])])
    typo = rapport.typographie
    assert typo["ligatures"] == 2  # œuvre, sœur
    assert typo["apostrophes_droites"] == 2  # L', qu'
    assert typo["guillemets"] == 2
    assert typo["insecables"] == 2
    # Les ligatures comptent comme marqueurs du français : œuvre, sœur, coûte.
    assert typo["mots_accentues"] == 3


def test_doublons_exacts_reperes(tmp_path):
    """Un corpus plein de doublons se mémorise au lieu de s'apprendre."""
    rapport = controler_corpus([_jsonl(tmp_path, [PROPRE, PROPRE, PROPRE, "Autre texte ici."])])
    assert rapport.n_doublons == 2
    assert "2 doublon" in str(rapport)


def test_documents_vides_comptes(tmp_path):
    rapport = controler_corpus([_jsonl(tmp_path, [PROPRE, "   ", ""])])
    # Les lignes vides du JSONL sont sautées ; « espaces seuls » compte pour vide.
    assert rapport.n_vides >= 1
    assert rapport.n_documents >= 2


def test_limite_de_documents(tmp_path):
    """Contrôler un échantillon d'un très gros fichier sans tout parcourir."""
    rapport = controler_corpus([_jsonl(tmp_path, [PROPRE] * 100)], documents_max=7)
    assert rapport.n_documents == 7


def test_corpus_texte_simple(tmp_path):
    """Le contrôle doit aussi accepter du .txt, pas seulement du JSONL."""
    chemin = tmp_path / "c.txt"
    chemin.write_text(PROPRE + "\n\n" + PROPRE, encoding="utf-8")
    rapport = controler_corpus([chemin])
    assert rapport.n_documents == 2


def test_corpus_introuvable(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        controler_corpus([tmp_path / "absent.jsonl"])


# --------------------------------------------------------------------------- #
# Affichage
# --------------------------------------------------------------------------- #


def test_le_rapport_est_lisible(tmp_path):
    rapport = controler_corpus([_jsonl(tmp_path, [PROPRE] * 3)])
    texte = str(rapport)
    for attendu in ("documents", "Traces de balisage", "Typographie française",
                    "mots accentués"):
        assert attendu in texte


def test_les_virgules_de_ponctuation_survivent_au_formatage(tmp_path):
    """Piège déjà tombé deux fois : formater une phrase entière puis y remplacer
    les virgules par des espaces avale aussi celles de la ponctuation."""
    rapport = controler_corpus([_jsonl(tmp_path, [PROPRE] * 20)])
    ligne = next(
        ligne for ligne in str(rapport).splitlines() if "longueur des documents" in ligne
    )
    assert "car., 1er décile" in ligne, f"Virgule mangée par le formatage : {ligne!r}"


def test_les_grands_nombres_sont_lisibles(tmp_path):
    from futo.controle import _milliers

    assert _milliers(1234567) == "1 234 567"
    assert _milliers(999) == "999"


# --------------------------------------------------------------------------- #
# Extraits
# --------------------------------------------------------------------------- #


def test_extraction_dexemples(tmp_path):
    """Les compteurs disent « propre », seule la lecture dit « lisible »."""
    extraits = extraire_echantillon([_jsonl(tmp_path, [PROPRE] * 5)], n=2, longueur=100)
    assert len(extraits) == 2
    assert all(len(e) <= 100 for e in extraits)
    assert "Lyon" in extraits[0]


def test_les_documents_trop_courts_ne_sont_pas_donnes_en_exemple(tmp_path):
    extraits = extraire_echantillon([_jsonl(tmp_path, ["Court.", PROPRE])], n=1)
    assert len(extraits) == 1
    assert "Lyon" in extraits[0]


def test_chaque_trace_a_une_description():
    """Un rapport doit être actionnable : chaque motif explique ce qu'il révèle."""
    for cle, expression, description in TRACES_DE_BALISAGE:
        assert cle and expression and description
        assert len(description) > 10, f"{cle} : description trop vague"
