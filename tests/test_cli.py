"""Tests de la ligne de commande."""

from __future__ import annotations

from futo.cli import main


def test_le_plafond_de_lecture_est_annonce(tmp_path, capsys):
    """Afficher la taille du fichier laissait croire que tout serait lu.

    Sur le vrai dump, « 7 321 453 631 octets » s'affichait alors que
    --octets-max limitait la lecture à 2 Go.
    """
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Le chat dort. " * 2000, encoding="utf-8")
    sortie = tmp_path / "tok.json"
    main([
        "tokenizer", "entrainer", "--corpus", str(corpus),
        "--sortie", str(sortie), "--vocab", "300", "--octets-max", "1000",
        "--silencieux",
    ])
    texte = capsys.readouterr().out
    assert "sur le disque" in texte
    assert "Texte réellement lu" in texte
    assert "1 000 octets" in texte
