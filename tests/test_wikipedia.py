"""Tests de la conversion d'un dump Wikipédia.

Aucun dump réel n'est nécessaire : le nettoyage du wikitexte se teste sur des
fragments écrits à la main, et le parcours XML sur un dump miniature fabriqué
ici. C'est ce qui permet à ces tests de tourner hors ligne, en quelques
millisecondes.

Ce que ces tests ne peuvent PAS garantir : le comportement sur les cas tordus
d'un vrai dump de plusieurs gigaoctets. Les surprises viendront de là.
"""

from __future__ import annotations

import bz2
import http.server
import json
import threading
from xml.sax.saxutils import escape

import pytest

from futo.wikipedia import (
    consigner_source,
    convertir_dump,
    lire_dump,
    nettoyer_wikitexte,
    telecharger_dump,
)

# --------------------------------------------------------------------------- #
# Nettoyage du wikitexte
# --------------------------------------------------------------------------- #


# Note sur les espaces : supprimer un modèle ou une balise laisse deux espaces
# consécutifs, que le nettoyage écrase ensuite en un seul. Les résultats
# attendus ci-dessous en tiennent compte — c'est voulu, un texte propre ne doit
# pas porter la trace de ce qui en a été retiré.
@pytest.mark.parametrize(
    "brut,attendu,ce_qui_est_teste",
    [
        ("Un texte simple.", "Un texte simple.", "texte sans balisage"),
        ("Le [[chat]] dort.", "Le chat dort.", "lien simple"),
        ("Le [[félin|chat]] dort.", "Le chat dort.", "lien avec libellé"),
        ("''italique'' et '''gras'''", "italique et gras", "gras et italique"),
        ("'''''les deux'''''", "les deux", "gras italique combinés"),
        ("Avant {{modele}} après.", "Avant après.", "modèle simple"),
        (
            "Avant {{Infobox|{{date|1|1|2000}}|x=y}} après.",
            "Avant après.",
            "modèles imbriqués — le piège des expressions régulières",
        ),
        ("Texte<ref>Une note</ref> suite.", "Texte suite.", "référence"),
        ('Texte<ref name="a" /> suite.', "Texte suite.", "référence auto-fermante"),
        ("Texte<!-- caché --> suite.", "Texte suite.", "commentaire HTML"),
        ("[[Fichier:photo.jpg|vignette|Une légende]]Texte.", "Texte.", "image"),
        ("[[Catégorie:Physique]]Texte.", "Texte.", "catégorie"),
        ("Texte [[Category:Physics]] suite.", "Texte suite.", "catégorie anglaise"),
        ("== Titre ==\nTexte.", "Titre\nTexte.", "titre de section"),
        ("=== Sous-titre ===\nTexte.", "Sous-titre\nTexte.", "sous-titre"),
        ("* un\n* deux", "un\ndeux", "liste à puces"),
        ("# un\n# deux", "un\ndeux", "liste numérotée"),
        ("[https://exemple.fr Un site]", "Un site", "lien externe avec libellé"),
        ("Texte &amp; suite", "Texte & suite", "entité HTML"),
        ("Texte<br />suite", "Textesuite", "balise HTML isolée"),
        ("{| class=\"x\"\n|a\n|}\nTexte.", "Texte.", "tableau"),
        ("<math>x^2</math> vaut.", "vaut.", "formule mathématique"),
    ],
    ids=lambda v: v if isinstance(v, str) and " " in v and len(v) < 60 else "",
)
def test_nettoyage(brut, attendu, ce_qui_est_teste):
    assert nettoyer_wikitexte(brut) == attendu, f"Échec sur : {ce_qui_est_teste}"


def test_modeles_profondement_imbriques():
    """Trois niveaux d'imbrication doivent disparaître entièrement.

    C'est précisément ce qu'une expression régulière ne sait pas faire : elle
    s'arrête au premier `}}` et laisse traîner les fermetures restantes.
    """
    brut = "Début {{a|{{b|{{c|x}}}}|y}} fin."
    assert nettoyer_wikitexte(brut) == "Début fin."
    assert "}}" not in nettoyer_wikitexte(brut)


def test_pas_despace_parasite_devant_le_point():
    """Supprimer un modèle ne doit pas laisser « avant . ».

    Mais la typographie française exige une espace insécable devant
    « ; : ! ? » et à l'intérieur des guillemets : elle doit survivre intacte.
    """
    assert nettoyer_wikitexte("Il commence avant {{date|43}}.") == "Il commence avant."
    assert nettoyer_wikitexte("Trois villes {{liste}}, dont Lyon.") == (
        "Trois villes, dont Lyon."
    )
    insecable = "\u00a0"
    for signe in (";", ":", "!", "?"):
        phrase = "Vraiment" + insecable + signe + " oui"
        assert nettoyer_wikitexte(phrase) == phrase, (
            f"L'espace insécable devant « {signe} » a été perdue."
        )
    guillemets = "«" + insecable + "Bonjour" + insecable + "»"
    assert nettoyer_wikitexte(guillemets) == guillemets


def test_les_apostrophes_delision_ne_sont_pas_mangees():
    """Le balisage de gras ne doit pas emporter l'apostrophe qui le précède.

    MediaWiki décompose une suite d'apostrophes : le balisage prend ce
    qu'il lui faut, le reste est du texte. Quatre apostrophes, c'est donc
    l'élision « L' » suivie du gras. Une règle trop grossière transformait
    « L'histoire » en « Lhistoire » — sur un corpus français, où l'élision
    est partout, la perte est loin d'être anodine.
    """
    q = "'"
    # La suite fermante rend elle aussi son excédent : on obtient une
    # apostrophe surnuméraire en fin de mot. Compromis assumé — mieux vaut
    # ce grain de bruit que la perte de l'apostrophe d'élision.
    assert nettoyer_wikitexte("L" + q * 4 + "Église" + q * 4) == "L" + q + "Église" + q
    # Le cas qui compte vraiment : l'élision en tête est préservée.
    assert nettoyer_wikitexte("L" + q * 4 + "Église" + q * 3).startswith("L" + q)
    assert nettoyer_wikitexte(q * 2 + "italique" + q * 2) == "italique"
    assert nettoyer_wikitexte(q * 3 + "gras" + q * 3) == "gras"
    assert nettoyer_wikitexte(q * 5 + "les deux" + q * 5) == "les deux"
    # Une apostrophe isolée est du texte, jamais du balisage.
    assert nettoyer_wikitexte("L" + q + "histoire") == "L" + q + "histoire"


def test_image_avec_lien_dans_la_legende():
    """Une légende d'image contient souvent des liens : tout doit partir."""
    brut = "[[Fichier:x.jpg|vignette|Vue de [[Paris]] en hiver]]Le texte."
    assert nettoyer_wikitexte(brut) == "Le texte."


def test_sections_finales_sont_coupees():
    """Références et liens externes sont retirés : très répétitifs, sans intérêt."""
    brut = (
        "Le corps de l'article.\n\n"
        "== Voir aussi ==\n* [[Autre article]]\n\n"
        "== Liens externes ==\n* Un lien\n"
    )
    resultat = nettoyer_wikitexte(brut)
    assert resultat == "Le corps de l'article."
    assert "Voir aussi" not in resultat


def test_le_francais_traverse_intact():
    """Accents, ligatures, apostrophes et guillemets doivent être préservés.

    C'est le point qui compte : le nettoyage ne doit jamais abîmer le français
    qu'il est censé extraire.
    """
    brut = (
        "L'''œuvre''' de [[Molière]] — « Le Malade imaginaire » — "
        "coûte 12 % de plus aujourd'hui, n'est-ce pas ?"
    )
    resultat = nettoyer_wikitexte(brut)
    for attendu in ("œuvre", "Molière", "«", "»", "coûte", "12 %",
                    "aujourd'hui", "n'est-ce pas"):
        assert attendu in resultat, f"« {attendu} » a été perdu : {resultat!r}"


def test_apostrophe_typographique_nest_pas_confondue_avec_litalique():
    """Le wikitexte marque l'italique par des apostrophes DROITES.

    L'apostrophe typographique du français doit donc traverser sans dommage.
    """
    assert nettoyer_wikitexte("L’œuvre de l’auteur.") == "L’œuvre de l’auteur."


def test_crochets_non_refermes_ne_font_pas_planter():
    """Un dump réel contient du balisage cassé : il ne doit rien casser."""
    for brut in ("Texte [[ cassé", "Texte {{ cassé", "Texte }} isolé", "[["):
        assert isinstance(nettoyer_wikitexte(brut), str)


def test_texte_vide():
    assert nettoyer_wikitexte("") == ""
    assert nettoyer_wikitexte("{{tout}}") == ""


# --------------------------------------------------------------------------- #
# Lecture du dump XML
# --------------------------------------------------------------------------- #

_GABARIT = """<?xml version="1.0" encoding="utf-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/" xml:lang="fr">
{pages}
</mediawiki>
"""


def _page(titre: str, texte: str, ns: str = "0", redirection: bool = False) -> str:
    """Fabrique une page de dump.

    Le texte est ÉCHAPPÉ, comme dans un vrai dump : le wikitexte contient des
    balises (<ref>, <math>) qui, laissées brutes, seraient interprétées comme
    du XML et casseraient la structure du document. MediaWiki les échappe en
    &lt;ref&gt; ; l'analyseur les rend ensuite telles quelles au nettoyeur.
    Ne pas échapper ici donnerait un banc d'essai plus permissif que la
    réalité, qui laisserait passer un défaut.
    """
    texte = escape(texte)
    titre = escape(titre)
    bloc_redirection = '    <redirect title="Ailleurs" />\n' if redirection else ""
    return (
        f"  <page>\n"
        f"    <title>{titre}</title>\n"
        f"    <ns>{ns}</ns>\n"
        f"{bloc_redirection}"
        f"    <revision>\n      <text>{texte}</text>\n    </revision>\n"
        f"  </page>"
    )


def _ecrire_dump(chemin, pages: list[str], comprime: bool = False):
    contenu = _GABARIT.format(pages="\n".join(pages))
    if comprime:
        chemin.write_bytes(bz2.compress(contenu.encode("utf-8")))
    else:
        chemin.write_text(contenu, encoding="utf-8")
    return chemin


LONG = "Un article de fond sur un sujet passionnant, écrit en français. " * 6


def test_lecture_dun_dump_minimal(tmp_path):
    dump = _ecrire_dump(tmp_path / "d.xml", [_page("Chat", LONG)])
    articles = list(lire_dump(dump))
    assert len(articles) == 1
    assert articles[0].titre == "Chat"
    assert "passionnant" in articles[0].texte


def test_dump_contenant_du_balisage_echappe(tmp_path):
    """Cas réel : le wikitexte d'un dump arrive échappé.

    &lt;ref&gt;Une note&lt;/ref&gt; doit traverser l'analyseur XML, redevenir
    une balise, puis être nettoyé. Un banc d'essai qui n'échapperait pas
    serait plus permissif que la réalité.
    """
    brut = (
        "L'''histoire''' de [[Lyon]] commence avant {{date|43|av. J.-C.}}. "
        "La ville a connu des bouleversements<ref>Une note</ref>. "
        "Son cœur historique est classé au patrimoine mondial. "
    ) * 3
    dump = _ecrire_dump(tmp_path / "d.xml", [_page("Lyon", brut)])
    articles = list(lire_dump(dump))

    assert len(articles) == 1
    texte = articles[0].texte
    assert "histoire de Lyon" in texte
    assert "cœur" in texte
    for indesirable in ("<ref>", "Une note", "{{", "}}", "[[", "]]", "'''"):
        assert indesirable not in texte, f"« {indesirable} » a survécu : {texte!r}"


def test_les_redirections_sont_ecartees(tmp_path):
    dump = _ecrire_dump(tmp_path / "d.xml", [
        _page("Article", LONG),
        _page("Renvoi", LONG, redirection=True),
    ])
    assert [a.titre for a in lire_dump(dump)] == ["Article"]


def test_seul_lespace_principal_est_retenu(tmp_path):
    """Discussions, modèles et catégories ne sont pas des articles."""
    dump = _ecrire_dump(tmp_path / "d.xml", [
        _page("Article", LONG, ns="0"),
        _page("Discussion:Article", LONG, ns="1"),
        _page("Modèle:Truc", LONG, ns="10"),
    ])
    assert [a.titre for a in lire_dump(dump)] == ["Article"]


def test_les_articles_trop_courts_sont_ecartes(tmp_path):
    """Une ébauche de deux lignes n'apprend rien et pollue le corpus."""
    dump = _ecrire_dump(tmp_path / "d.xml", [
        _page("Long", LONG),
        _page("Court", "Trop court."),
    ])
    assert [a.titre for a in lire_dump(dump, longueur_min=200)] == ["Long"]


def test_dump_compresse(tmp_path):
    """Les dumps officiels arrivent en .bz2 : il faut les lire tels quels."""
    dump = _ecrire_dump(tmp_path / "d.xml.bz2", [_page("Chat", LONG)], comprime=True)
    assert len(list(lire_dump(dump))) == 1


def test_dump_absent_leve_une_erreur_actionnable(tmp_path):
    with pytest.raises(FileNotFoundError, match="futo data telecharger"):
        list(lire_dump(tmp_path / "absent.xml"))


# --------------------------------------------------------------------------- #
# Conversion complète
# --------------------------------------------------------------------------- #


def test_conversion_produit_du_jsonl_lisible(tmp_path):
    """Le fichier produit doit être exactement ce qu'attend `futo data preparer`."""
    dump = _ecrire_dump(tmp_path / "d.xml", [
        _page("Chat", LONG), _page("Chien", LONG),
    ])
    sortie = tmp_path / "corpus.jsonl"
    resultat = convertir_dump(dump, sortie, journal=None)

    assert resultat.n_articles == 2
    assert resultat.n_caracteres > 0

    lignes = sortie.read_text(encoding="utf-8").strip().split("\n")
    assert len(lignes) == 2
    for ligne in lignes:
        objet = json.loads(ligne)
        assert "text" in objet and objet["text"].strip()
        assert "titre" in objet


def test_le_jsonl_produit_est_relisible_par_le_pipeline(tmp_path, tokenizer):
    """Le vrai test d'intégration : du dump jusqu'aux shards."""
    from futo.data import preparer_corpus

    dump = _ecrire_dump(tmp_path / "d.xml", [
        _page(f"Article {i}", LONG) for i in range(12)
    ])
    jsonl = tmp_path / "corpus.jsonl"
    convertir_dump(dump, jsonl, journal=None)

    resultat = preparer_corpus(
        [jsonl], tokenizer, tmp_path / "prep", tokens_par_shard=5_000,
        fraction_val=0.0, separateur="jsonl", journal=None,
    )
    assert resultat.n_documents == 12
    assert resultat.n_tokens > 0


def test_limite_darticles(tmp_path):
    """Pouvoir s'arrêter tôt permet d'essayer sans traiter tout le dump."""
    dump = _ecrire_dump(tmp_path / "d.xml", [
        _page(f"A{i}", LONG) for i in range(10)
    ])
    resultat = convertir_dump(dump, tmp_path / "c.jsonl", articles_max=3, journal=None)
    assert resultat.n_articles == 3


def test_dump_sans_article_est_signale(tmp_path):
    dump = _ecrire_dump(tmp_path / "d.xml", [_page("Discussion", LONG, ns="1")])
    with pytest.raises(ValueError, match="Aucun article"):
        convertir_dump(dump, tmp_path / "c.jsonl", journal=None)


def test_aucun_fichier_partiel_ne_subsiste(tmp_path):
    """Un plantage en cours de route ne doit pas laisser un corpus tronqué
    qu'on prendrait ensuite pour un corpus complet."""
    dump = _ecrire_dump(tmp_path / "d.xml", [_page("A", LONG)])
    sortie = tmp_path / "c.jsonl"
    convertir_dump(dump, sortie, journal=None)
    assert sortie.exists()
    assert not list(tmp_path.glob("*.partiel"))


# --------------------------------------------------------------------------- #
# Traçabilité des sources
# --------------------------------------------------------------------------- #


def test_la_source_est_consignee(tmp_path):
    """Exigé par le règlement européen sur l'IA, et impossible à reconstituer
    après coup."""
    dump = _ecrire_dump(tmp_path / "frwiki.xml", [_page("A", LONG)])
    resultat = convertir_dump(dump, tmp_path / "c.jsonl", journal=None)

    sources = tmp_path / "SOURCES.md"
    consigner_source(sources, "Wikipédia FR", dump, resultat)
    contenu = sources.read_text(encoding="utf-8")
    assert "Wikipédia FR" in contenu
    assert "frwiki.xml" in contenu
    assert "CC BY-SA" in contenu
    assert "| Date | Source |" in contenu

    # Un second ajout complète le tableau sans réécrire l'en-tête.
    consigner_source(sources, "Autre corpus", dump, resultat, licence="CC0")
    contenu = sources.read_text(encoding="utf-8")
    assert contenu.count("| Date | Source |") == 1
    assert "CC0" in contenu


# --------------------------------------------------------------------------- #
# Téléchargement
# --------------------------------------------------------------------------- #


class _ServeurAvecRange(http.server.BaseHTTPRequestHandler):
    """Serveur minimal qui gère l'en-tête Range, pour éprouver la reprise.

    `SimpleHTTPRequestHandler` ne le gère pas : sans cela, on ne testerait que
    le cas facile, celui où rien ne se coupe.
    """

    contenu = b""

    def do_GET(self):  # noqa: N802 (nom imposé par la bibliothèque standard)
        debut = 0
        plage = self.headers.get("Range")
        if plage and plage.startswith("bytes="):
            debut = int(plage.removeprefix("bytes=").split("-")[0])
        if debut >= len(self.contenu):
            self.send_error(416)
            return
        morceau = self.contenu[debut:]
        self.send_response(206 if debut else 200)
        self.send_header("Content-Length", str(len(morceau)))
        if debut:
            self.send_header(
                "Content-Range",
                f"bytes {debut}-{len(self.contenu) - 1}/{len(self.contenu)}",
            )
        self.end_headers()
        self.wfile.write(morceau)

    def log_message(self, *args):
        pass  # silence pendant les tests


@pytest.fixture
def serveur(monkeypatch):
    """Sert un contenu fixe sur localhost, le temps d'un test."""
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    _ServeurAvecRange.contenu = bz2.compress(b"contenu de dump " * 5_000)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ServeurAvecRange)
    fil = threading.Thread(target=httpd.serve_forever, daemon=True)
    fil.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/dump.bz2"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_telechargement_complet(serveur, tmp_path):
    cible = tmp_path / "dump.xml.bz2"
    resultat = telecharger_dump(serveur, cible, journal=None)
    assert resultat == cible
    assert cible.read_bytes() == _ServeurAvecRange.contenu
    # Aucun fichier partiel ne doit subsister après un téléchargement réussi.
    assert not list(tmp_path.glob("*.partiel"))


def test_reprise_apres_coupure(serveur, tmp_path):
    """Le cas qui compte : 7 Gio sur une liaison domestique, ça coupe.

    On simule une coupure en écrivant à la main un fichier partiel, puis on
    vérifie que la reprise complète le fichier au lieu de tout recommencer.
    """
    cible = tmp_path / "dump.xml.bz2"
    partiel = cible.with_suffix(cible.suffix + ".partiel")
    coupure = len(_ServeurAvecRange.contenu) // 3
    partiel.write_bytes(_ServeurAvecRange.contenu[:coupure])

    telecharger_dump(serveur, cible, journal=None)
    assert cible.read_bytes() == _ServeurAvecRange.contenu


def test_fichier_deja_complet_nest_pas_retelecharge(serveur, tmp_path):
    """Relancer la commande ne doit pas refaire une heure de téléchargement."""
    cible = tmp_path / "dump.xml.bz2"
    cible.write_bytes(b"deja la")
    telecharger_dump(serveur, cible, journal=None)
    assert cible.read_bytes() == b"deja la"  # intact, non réécrit


def test_adresse_invalide_leve_une_erreur_actionnable(tmp_path, monkeypatch):
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    with pytest.raises(ValueError, match="Téléchargement impossible"):
        telecharger_dump("http://127.0.0.1:1/absent.bz2", tmp_path / "d.bz2", journal=None)


def test_le_dump_telecharge_est_lisible(serveur, tmp_path):
    """Chaîne complète : télécharger, puis convertir, sans intervention."""
    _ServeurAvecRange.contenu = bz2.compress(
        _GABARIT.format(pages=_page("Chat", LONG)).encode("utf-8")
    )
    cible = tmp_path / "dump.xml.bz2"
    telecharger_dump(serveur, cible, journal=None)
    resultat = convertir_dump(cible, tmp_path / "corpus.jsonl", journal=None)
    assert resultat.n_articles == 1
