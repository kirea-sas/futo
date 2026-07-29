"""Tests de la documentation.

La documentation est un livrable comme un autre : si un bloc de commandes du
README ne peut pas être copié-collé, il est faux, et il fait perdre du temps à
la première personne qui essaie le projet.

Ces tests ne lisent aucun fichier hors du dépôt et ne touchent pas au réseau.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent

# Les dossiers à ne jamais parcourir. `.venv` compte : le README conseille de
# créer l'environnement virtuel DANS le dépôt, et site-packages regorge de
# fichiers Markdown de bibliothèques tierces. Sans cette exclusion, la suite
# testait la documentation de PyTorch — 51 tests de plus, et le risque d'un
# échec incompréhensible sur un README qui ne nous appartient pas.
# Signalé par un run sur un Mac où le venv vivait dans le dépôt.
DOSSIERS_IGNORES = {
    ".git", ".venv", "venv", "env", "node_modules", "site-packages",
    "build", "dist", ".tox", ".pytest_cache", ".ruff_cache", "__pycache__",
}
DOCUMENTS = sorted(
    p for p in RACINE.rglob("*.md")
    if not DOSSIERS_IGNORES & set(p.parts) and not any(
        partie.endswith((".egg-info", ".dist-info")) for partie in p.parts
    )
)


def _blocs_shell(texte: str) -> list[tuple[int, str]]:
    """Renvoie les lignes des blocs ```bash, avec leur numéro dans le fichier."""
    lignes: list[tuple[int, str]] = []
    dedans = False
    for numero, ligne in enumerate(texte.splitlines(), 1):
        if ligne.startswith("```"):
            dedans = ligne.startswith(("```bash", "```sh", "```console"))
            continue
        if dedans:
            lignes.append((numero, ligne))
    return lignes


def test_il_y_a_bien_des_documents():
    assert DOCUMENTS, "Aucun document Markdown trouvé."
    noms = {p.name for p in DOCUMENTS}
    assert {"README.md", "ROADMAP.md"} <= noms


def test_aucun_document_etranger_nest_ramasse():
    """La collecte ne doit ramasser QUE les documents du dépôt.

    Un environnement virtuel créé dans le dépôt — ce que le README conseille —
    apporte des dizaines de fichiers Markdown de bibliothèques tierces. Les
    tester n'a aucun sens, gonfle le nombre de tests, et peut faire échouer la
    suite sur un README qui ne nous appartient pas.
    """
    etrangers = [
        str(d.relative_to(RACINE)) for d in DOCUMENTS
        if DOSSIERS_IGNORES & set(d.parts)
    ]
    assert not etrangers, f"Documents hors du dépôt ramassés : {etrangers}"
    # Le dépôt est petit : au-delà d'une vingtaine, c'est qu'on ratisse trop.
    assert len(DOCUMENTS) < 20, (
        f"{len(DOCUMENTS)} documents collectés — la collecte ratisse trop large."
    )


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: str(p.relative_to(RACINE)))
def test_commentaires_shell_sans_apostrophe(document: Path):
    """Une apostrophe dans un commentaire shell bloque un copier-coller sous zsh.

    zsh n'active pas `interactive_comments` par défaut dans toutes les
    configurations. Quand il est absent, le « # » n'ouvre pas un commentaire :
    l'apostrophe de « ce qu'il a appris » ouvre alors une chaîne que rien ne
    ferme, et le terminal reste bloqué sur l'invite « quote> » sans que rien ne
    s'exécute. Le lecteur croit que le projet est cassé.

    Les commentaires des blocs de commandes s'écrivent donc sans apostrophe.
    """
    fautifs = [
        (numero, ligne.strip())
        for numero, ligne in _blocs_shell(document.read_text(encoding="utf-8"))
        if ligne.lstrip().startswith("#") and ("'" in ligne or "’" in ligne)
    ]
    assert not fautifs, (
        f"{document.relative_to(RACINE)} : apostrophe dans un commentaire shell "
        f"— un copier-coller sous zsh bloquerait sur « quote> » :\n"
        + "\n".join(f"  ligne {n} : {c}" for n, c in fautifs)
    )


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: str(p.relative_to(RACINE)))
def test_guillemets_apparies_dans_les_blocs_shell(document: Path):
    """Chaque ligne de commande doit avoir ses guillemets appariés.

    Vérification volontairement simple — on compte les guillemets hors
    commentaire, en ignorant les lignes qui se poursuivent (« \\ » final), car
    une chaîne peut légitimement s'y étendre.
    """
    fautifs = []
    for numero, ligne in _blocs_shell(document.read_text(encoding="utf-8")):
        nue = ligne.strip()
        if not nue or nue.startswith("#") or nue.endswith("\\"):
            continue
        for signe in ("'", '"'):
            if nue.count(signe) % 2 != 0:
                fautifs.append((numero, nue, signe))
    assert not fautifs, (
        f"{document.relative_to(RACINE)} : guillemets non appariés :\n"
        + "\n".join(f"  ligne {n} ({s}) : {c}" for n, c, s in fautifs)
    )


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: str(p.relative_to(RACINE)))
def test_liens_internes_pointent_vers_des_fichiers_existants(document: Path):
    """Un lien mort dans le README est une promesse non tenue."""
    texte = document.read_text(encoding="utf-8")
    casses = []
    for cible in re.findall(r"\[[^\]]*\]\(([^)]+)\)", texte):
        if cible.startswith(("http://", "https://", "#", "mailto:")):
            continue
        chemin = (document.parent / cible.split("#")[0]).resolve()
        if not chemin.exists():
            casses.append(cible)
    assert not casses, f"{document.relative_to(RACINE)} : liens cassés {casses}"


def test_le_readme_cite_toutes_les_configurations():
    """Toute configuration livrée doit être mentionnée quelque part dans la
    documentation : une configuration que personne ne découvre n'existe pas."""
    readme = (RACINE / "README.md").read_text(encoding="utf-8")
    docs = "\n".join(p.read_text(encoding="utf-8") for p in DOCUMENTS)
    for chemin in sorted((RACINE / "configs").glob("futo-*.yaml")):
        nom = chemin.stem
        assert nom in docs, f"{nom} n'est cité dans aucun document."
        assert nom in readme or "docs/" in readme, f"{nom} est introuvable depuis le README."


def test_la_licence_des_poids_est_tranchee_et_coherente():
    """La licence des poids doit être décidée, et dite partout pareil.

    Une licence tranchée mais non écrite ne vaut rien, et une licence écrite
    différemment selon les documents vaut pire que rien. Le choix — CC BY-SA 4.0,
    reprise de celle de Wikipédia — doit apparaître dans le README comme dans la
    carte du modèle, et aucun document ne doit encore le présenter comme ouvert.
    """
    readme = (RACINE / "README.md").read_text(encoding="utf-8")
    carte = (RACINE / "docs" / "CARTE-DU-MODELE.md").read_text(encoding="utf-8")

    for nom, contenu in (("README.md", readme), ("CARTE-DU-MODELE.md", carte)):
        assert "CC BY-SA 4.0" in contenu, f"{nom} ne dit pas la licence des poids."

    # Le code, lui, reste sous Apache 2.0 : les deux doivent coexister sans
    # que l'un soit pris pour l'autre.
    assert "Apache 2.0" in readme

    # Plus aucune formulation en suspens.
    for nom, contenu in (("README.md", readme), ("CARTE-DU-MODELE.md", carte),
                         ("DONNEES.md", (RACINE / "docs" / "DONNEES.md").read_text(encoding="utf-8"))):
        for en_suspens in ("à trancher selon", "non tranché ;"):
            assert en_suspens not in contenu, (
                f"{nom} présente encore la licence des poids comme ouverte "
                f"(« {en_suspens} »), alors qu'elle est décidée."
            )

    # La valeur par défaut du journal des sources doit suivre la même licence.
    import inspect

    from futo.wikipedia import consigner_source

    signature = inspect.signature(consigner_source)
    assert signature.parameters["licence"].default == "CC BY-SA 4.0"


def test_commandes_du_readme_existent_dans_la_cli():
    """Les sous-commandes citées dans le README doivent exister réellement.

    Empêche la dérive classique : on renomme une sous-commande et la
    documentation continue d'annoncer l'ancienne.
    """
    from futo.cli import construire_analyseur

    analyseur = construire_analyseur()
    actions = [
        a for a in analyseur._actions if hasattr(a, "choices") and a.choices
    ]
    connues = set()
    for action in actions:
        connues.update(action.choices)

    readme = (RACINE / "README.md").read_text(encoding="utf-8")
    citees = set(re.findall(r"^\s*futo ([a-z]+)", readme, flags=re.MULTILINE))
    inconnues = citees - connues
    assert not inconnues, (
        f"Le README cite des sous-commandes inexistantes : {sorted(inconnues)}. "
        f"Sous-commandes réelles : {sorted(connues)}."
    )
