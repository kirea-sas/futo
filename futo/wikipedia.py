"""Conversion d'un dump Wikipédia en corpus d'entraînement.

Wikipédia est le meilleur point de départ pour Futo : du français relu, à peu
près à la bonne taille pour `futo-mac`, et surtout **déjà propre** — ce qui
permet de se passer du pipeline de filtrage que réclame le web brut.

Le travail consiste à sortir du texte lisible d'un dump XML de plusieurs
gigaoctets. Deux difficultés, traitées ici :

**La taille.** Un dump francophone compressé pèse plusieurs gigaoctets, et bien
davantage décompressé. On ne le charge donc jamais en mémoire : lecture en flux
avec `iterparse`, et libération de chaque élément dès qu'il est traité. Sans ce
`clear()`, l'arbre XML s'accumule et le processus finit par saturer la mémoire.

**Le wikitexte.** Les articles ne sont pas du texte mais un langage de balisage :
modèles imbriqués `{{...}}`, tableaux `{|...|}`, liens `[[a|b]]`, références,
catégories. Le nettoyage ci-dessous est volontairement conservateur — mieux vaut
laisser passer un fragment de balisage que d'amputer une phrase. Chaque règle a
son test dans `tests/test_wikipedia.py`.

Sur la licence : le contenu de Wikipédia est sous CC BY-SA, ce qui impose
l'attribution et le partage à l'identique. L'effet sur des poids entraînés n'est
pas tranché juridiquement. À décider **avant** l'entraînement — voir
`docs/DONNEES.md`. La commande consigne systématiquement la source dans
`data/SOURCES.md` : c'est aussi ce qu'exige le règlement européen sur l'IA.
"""

from __future__ import annotations

import bz2
import gzip
import html
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "URL_DUMP_FR",
    "telecharger_dump",
    "nettoyer_wikitexte",
    "lire_dump",
    "convertir_dump",
    "ResultatConversion",
]


# --------------------------------------------------------------------------- #
# Nettoyage du wikitexte
# --------------------------------------------------------------------------- #

# Balises dont on jette le contenu : ce sont des notes, des formules ou des
# éléments de mise en page, jamais de la prose.
_BALISES_A_VIDER = (
    "ref", "math", "chem", "timeline", "gallery", "imagemap",
    "score", "graph", "mapframe", "templatestyles",
)

_MOTIF_COMMENTAIRE = re.compile(r"<!--.*?-->", re.DOTALL)
_MOTIF_BALISES_VIDES = re.compile(
    r"<(" + "|".join(_BALISES_A_VIDER) + r")\b[^>]*?/>", re.IGNORECASE
)
_MOTIF_BALISES_PLEINES = re.compile(
    r"<(" + "|".join(_BALISES_A_VIDER) + r")\b[^>]*?>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_MOTIF_BALISE_HTML = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*\b[^>]*/?>")
_MOTIF_LIEN_EXTERNE = re.compile(r"\[(?:https?:|ftp:|//)[^\s\]]+\s*([^\]]*)\]")
# Titres de section. Les espacements admis sont HORIZONTAUX seulement.
# Avec un espacement gourmand incluant le saut de ligne, la fin du motif
# avalait le retour à la ligne et le titre suivant avalait celui de la ligne
# vide : « == Biographie == » suivi de « === Enfance === » ressortait collé,
# en « BiographieEnfance ». Relevé sur un vrai article de frwiki.
_MOTIF_TITRE = re.compile(r"^[ \t]*(={2,6})[ \t]*(.*?)[ \t]*\1[ \t]*$", re.MULTILINE)
_MOTIF_APOSTROPHES = re.compile(r"'{2,}")


_MOTIF_PUCE = re.compile(r"^[*#:;]+\s*", re.MULTILINE)
_MOTIF_ESPACES = re.compile(r"[ \t]{2,}")
_MOTIF_LIGNES_VIDES = re.compile(r"\n{3,}")
# Espace parasite devant un point ou une virgule, laissée par la suppression
# d'un modèle ou d'une balise : « commence avant {{date}}. » donnait
# « commence avant . ». Volontairement limité au point et à la virgule :
# le français demande au contraire une espace insécable devant ; : ! ? »
# et il ne faut surtout pas y toucher.
_MOTIF_ESPACE_AVANT_POINT = re.compile(r"[ \t]+([.,])")

# Filet de sécurité, appliqué en dernier. Une ligne de français ne commence
# jamais par « | » ni par « ! » : ce sont des lignes de tableau dont le
# « {| » d'ouverture a été perdu. Et les débris de balisage — accolades,
# crochets doubles — ne sont jamais du texte. On les retire plutôt que de
# les laisser au modèle, qui apprendrait à les reproduire.
_MOTIF_LIGNE_TABLEAU = re.compile(r"^[|!].*$", re.MULTILINE)
_MOTIF_DEBRIS = re.compile(r"\{\{|\}\}|\{\||\|\}|\[\[|\]\]")

# Préfixes de liens qu'on supprime entièrement, avec leur légende : ce sont des
# médias ou des métadonnées, pas du texte d'article. Les formes anglaises
# coexistent avec les françaises dans les dumps.
_PREFIXES_A_SUPPRIMER = (
    "fichier:", "file:", "image:", "média:", "media:",
    "catégorie:", "category:",
)

# Sections de fin d'article : listes de liens et de références, sans intérêt
# pour un modèle de langue et très répétitives d'un article à l'autre.
_SECTIONS_FINALES = {
    "voir aussi", "notes et références", "notes", "références", "liens externes",
    "bibliographie", "annexes", "articles connexes", "sources", "pour approfondir",
}


# Modèles dont le CONTENU est du texte : les supprimer casse la phrase.
# « né le {{date|11|novembre|1866}} à Moulins » devenait « né le à Moulins » —
# du français agrammatical, que le modèle apprendrait tel quel. Relevé sur
# frwiki dans l'article Antoine Meillet.
#
# La liste est volontairement courte : seulement les modèles très fréquents
# dont on sait rendre le texte sans risque. Tous les autres restent supprimés.
_MOTIF_MODELE_SIMPLE = re.compile(r"\{\{([^{}]*)\}\}")


def _rendre_modele(correspondance: re.Match) -> str:
    """Rend en texte les quelques modèles qui portent du sens, supprime le reste.

    Appliqué aux modèles les plus INTÉRIEURS d'abord, puis répété : un modèle
    imbriqué dans un autre est donc rendu avant que l'extérieur ne soit traité.
    """
    contenu = correspondance.group(1)
    morceaux = [m.strip() for m in contenu.split("|")]
    tete = morceaux[0]
    # Les arguments nommés (clé=valeur) sont de la mise en forme, pas du texte.
    positionnels = [m for m in morceaux[1:] if "=" not in m and m]

    # {{formatnum:12345}} et consorts : le nom porte la valeur après « : ».
    if ":" in tete:
        nom, _, valeur = tete.partition(":")
        nom = nom.strip().lower()
        if nom in {"formatnum", "nombre", "unité", "unite"}:
            return valeur.strip()
        return ""

    nom = tete.lower()

    if nom in {"date", "date-", "date de naissance", "date de décès"}:
        return " ".join(positionnels[:3])
    if nom in {"s", "s-", "-s", "s2", "siècle", "siecle"}:
        # {{s-|XX|e}} vaut « XXe siècle » ; l'exposant est « e » par défaut.
        if not positionnels:
            return ""
        exposant = positionnels[1] if len(positionnels) > 1 else "e"
        return f"{positionnels[0]}{exposant} siècle"
    if nom in {"unité", "unite", "nombre", "nb", "num", "formatnum"}:
        return "\u00a0".join(positionnels)
    if nom in {"lang", "langue", "lang-en", "en"}:
        return positionnels[-1] if positionnels else ""
    if nom in {"citation", "guil", "citation bloc"}:
        return f"«\u00a0{positionnels[0]}\u00a0»" if positionnels else ""
    if nom in {"nobr", "nowrap", "abréviation", "abbr"}:
        return positionnels[0] if positionnels else ""

    return ""  # tout le reste : supprimé, comme avant


def _rendre_modeles_connus(texte: str) -> str:
    """Traite les modèles de l'intérieur vers l'extérieur, jusqu'à épuisement."""
    for _ in range(12):  # borne de sûreté : un wikitexte cassé ne doit pas boucler
        nouveau = _MOTIF_MODELE_SIMPLE.sub(_rendre_modele, texte)
        if nouveau == texte:
            return nouveau
        texte = nouveau
    return texte


def _reduire_apostrophes(correspondance: re.Match) -> str:
    """Retire le balisage de gras et d'italique sans manger les apostrophes.

    MediaWiki code l'italique par deux apostrophes et le gras par trois. Une
    suite plus longue se décompose : le balisage prend ce qu'il lui faut, le
    reste est du texte. C'est le cas de `L''''Église''''` — quatre apostrophes,
    soit l'élision « L' » suivie du gras.

    Supprimer bêtement toute suite de deux à cinq apostrophes transformait
    « L'histoire » en « Lhistoire ». Sur un corpus français, où l'élision est
    partout, cette perte est loin d'être anodine.

    Règle appliquée : 2, 3 et 5 apostrophes sont du balisage pur ; 4 en rendent
    une au texte ; **au-delà de 5, on n'en rend qu'une seule**. MediaWiki en
    rendrait n − 5, mais deux apostrophes consécutives ne sont jamais du
    français : une suite de huit laissait ainsi trois apostrophes en plein
    texte, repérées comme balisage résiduel sur le vrai dump.

    Conséquence assumée : dans `L''''Église''''`, la suite FERMANTE rend elle
    aussi son excédent, et l'on obtient `L'Église'`. On préfère cette
    apostrophe surnuméraire en fin de mot à la perte de celle de l'élision :
    la première est un grain de bruit, la seconde une faute de français
    répétée des milliers de fois dans le corpus.
    """
    n = len(correspondance.group(0))
    if n in (2, 3, 5):
        return ""
    if n == 4:
        return "'"
    return "'"  # 6 et plus : une seule, jamais une rafale


def _traiter_liens_internes(texte: str) -> str:
    """Réduit `[[cible|libellé]]` à son libellé, en gérant l'imbrication.

    Les liens vers des médias et des catégories sont supprimés entièrement,
    légende comprise : une légende d'image isolée n'a pas de contexte, et les
    catégories ne sont pas de la prose.
    """
    resultat: list[str] = []
    i = 0
    n = len(texte)

    while i < n:
        if texte.startswith("[[", i):
            # Trouver le ]] correspondant, en comptant les imbrications.
            profondeur = 1
            j = i + 2
            while j < n and profondeur > 0:
                if texte.startswith("[[", j):
                    profondeur += 1
                    j += 2
                elif texte.startswith("]]", j):
                    profondeur -= 1
                    j += 2
                else:
                    j += 1
            if profondeur != 0:  # crochets non refermés : on laisse tel quel
                resultat.append(texte[i])
                i += 1
                continue

            contenu = texte[i + 2 : j - 2]
            debut = contenu.split(":", 1)[0].strip().lower() + ":"
            if debut in _PREFIXES_A_SUPPRIMER:
                pass  # média ou catégorie : on jette tout
            else:
                # `[[a|b]]` donne « b » ; `[[a]]` donne « a ». Les liens de
                # médias imbriqués dans la légende ont déjà été traités par la
                # récursion ci-dessous.
                morceaux = contenu.split("|")
                libelle = morceaux[-1] if len(morceaux) > 1 else morceaux[0]
                resultat.append(_traiter_liens_internes(libelle))
            i = j
        else:
            resultat.append(texte[i])
            i += 1
    return "".join(resultat)


def _supprimer_blocs_wiki(texte: str) -> str:
    """Supprime modèles `{{…}}` et tableaux `{|…|}` en une seule passe.

    Deux défauts d'une version antérieure, tous deux révélés par un vrai dump
    et invisibles sur des fragments écrits à la main :

    **Un bloc jamais refermé emportait tout le reste de l'article.** Traiter les
    tableaux puis les modèles avec un simple compteur de profondeur revient à
    supposer que le wikitexte est bien formé. Il ne l'est pas : sur des millions
    d'articles édités à la main, il y a des `{{` orphelins. Le compteur ne
    revenait alors jamais à zéro et la fin du document disparaissait.

    **Modèles et tableaux s'imbriquent l'un dans l'autre.** Les traiter en deux
    passes séparées coupe les paires croisées et laisse des fermetures
    orphelines — les `}}` et `|}` isolés qu'on retrouvait dans le texte final.

    D'où cette passe unique, avec une pile qui sait ce qu'elle a ouvert, et une
    règle prudente : **on ne supprime que les blocs effectivement refermés**.
    Un bloc orphelin est laissé tel quel ; le filet de sécurité en fin de
    nettoyage se charge de ses débris. Mieux vaut un `{{` résiduel qu'un
    paragraphe manquant.
    """
    plages: list[tuple[int, int]] = []  # blocs de plus haut niveau, bien fermés
    pile: list[tuple[int, str]] = []
    i, n = 0, len(texte)

    while i < n:
        if texte.startswith("{{", i):
            pile.append((i, "}}"))
            i += 2
        elif texte.startswith("{|", i):
            pile.append((i, "|}"))
            i += 2
        elif pile and texte.startswith(pile[-1][1], i):
            debut, fermeture = pile.pop()
            i += len(fermeture)
            if not pile:  # bloc complet, au plus haut niveau
                plages.append((debut, i))
        else:
            i += 1

    if not plages:
        return texte

    morceaux: list[str] = []
    precedent = 0
    for debut, fin in plages:
        morceaux.append(texte[precedent:debut])
        precedent = fin
    morceaux.append(texte[precedent:])
    return "".join(morceaux)


def _couper_sections_finales(texte: str) -> str:
    """Tronque l'article à la première section de références ou de liens."""
    lignes = texte.split("\n")
    for indice, ligne in enumerate(lignes):
        nue = ligne.strip()
        if nue.startswith("==") and nue.endswith("=="):
            titre = nue.strip("= ").strip().lower()
            if titre in _SECTIONS_FINALES:
                return "\n".join(lignes[:indice])
    return texte


def nettoyer_wikitexte(brut: str) -> str:
    """Transforme du wikitexte en prose lisible.

    L'ordre des opérations compte : on retire d'abord ce qui peut contenir
    n'importe quoi (commentaires, balises, tableaux, modèles), et seulement
    ensuite on traite les liens et la typographie.
    """
    texte = brut

    # 1. Commentaires et balises dont le contenu est à jeter.
    texte = _MOTIF_COMMENTAIRE.sub("", texte)
    texte = _MOTIF_BALISES_PLEINES.sub("", texte)
    texte = _MOTIF_BALISES_VIDES.sub("", texte)

    # 2. Les modèles dont le contenu est du texte — dates, siècles, nombres,
    #    unités — sont RENDUS ; les autres disparaissent. Traités de l'intérieur
    #    vers l'extérieur, avant la suppression en bloc.
    texte = _rendre_modeles_connus(texte)

    # 3. Ce qui reste de modèles et de tableaux, en UNE passe : ils s'imbriquent
    #    l'un dans l'autre, et les traiter séparément coupe les paires croisées.
    texte = _supprimer_blocs_wiki(texte)

    # 4. Sections de fin (références, liens externes…), tant que les titres
    #    sont encore reconnaissables.
    texte = _couper_sections_finales(texte)

    # 5. Liens internes, puis externes.
    texte = _traiter_liens_internes(texte)
    texte = _MOTIF_LIEN_EXTERNE.sub(r"\1", texte)

    # 6. Typographie du wikitexte.
    texte = _MOTIF_TITRE.sub(r"\2", texte)
    texte = _MOTIF_APOSTROPHES.sub(_reduire_apostrophes, texte)
    texte = _MOTIF_PUCE.sub("", texte)

    # 7. Ce qui reste de HTML, puis les entités.
    texte = _MOTIF_BALISE_HTML.sub("", texte)
    texte = html.unescape(texte)

    # 8. Filet de sécurité : ce qui a survécu au balisage n'est pas du français.
    texte = _MOTIF_LIGNE_TABLEAU.sub("", texte)
    texte = _MOTIF_DEBRIS.sub("", texte)

    # 9. Mise au propre finale.
    texte = _MOTIF_ESPACES.sub(" ", texte)
    texte = _MOTIF_ESPACE_AVANT_POINT.sub(r"\1", texte)
    lignes = [ligne.strip() for ligne in texte.split("\n")]
    texte = "\n".join(lignes)
    texte = _MOTIF_LIGNES_VIDES.sub("\n\n", texte)
    return texte.strip()


# --------------------------------------------------------------------------- #
# Téléchargement
# --------------------------------------------------------------------------- #

# L'adresse officielle des dumps francophones. « latest » suit toujours la
# dernière version publiée — pratique, mais cela veut dire que deux
# téléchargements à quelques semaines d'écart ne donnent pas le même corpus.
# D'où la trace écrite dans data/SOURCES.md : sans elle, impossible de dire
# plus tard sur quoi un modèle a été entraîné.
URL_DUMP_FR = (
    "https://dumps.wikimedia.org/frwiki/latest/"
    "frwiki-latest-pages-articles.xml.bz2"
)


def telecharger_dump(
    url: str = URL_DUMP_FR,
    destination: str | Path = "data/brut/frwiki-latest-pages-articles.xml.bz2",
    journal=None,
) -> Path:
    """Récupère un dump, avec reprise si le téléchargement a été interrompu.

    Plusieurs gigaoctets sur une liaison domestique, c'est long, et une coupure
    en cours de route est la règle plutôt que l'exception. Le fichier est donc
    écrit sous un nom temporaire, et une reprise repart de l'octet où l'on
    s'était arrêté grâce à l'en-tête HTTP `Range`. Un dump déjà complet n'est
    pas retéléchargé.
    """
    import urllib.error
    import urllib.request

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partiel = destination.with_suffix(destination.suffix + ".partiel")

    def dire(message: str) -> None:
        if journal is not None:
            journal(message)

    if destination.exists():
        dire(f"  déjà présent : {destination} "
             f"({destination.stat().st_size / 2**30:.2f} Gio) — rien à faire.")
        return destination

    deja = partiel.stat().st_size if partiel.exists() else 0
    requete = urllib.request.Request(url, headers={"User-Agent": "futo/0.1"})
    if deja:
        dire(f"  reprise à {deja / 2**30:.2f} Gio")
        requete.add_header("Range", f"bytes={deja}-")

    try:
        reponse = urllib.request.urlopen(requete, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and deja:  # déjà tout téléchargé
            partiel.replace(destination)
            return destination
        raise ValueError(
            f"Téléchargement impossible ({e.code} {e.reason}) depuis {url}.\n"
            f"Vérifiez l'adresse sur https://dumps.wikimedia.org/frwiki/"
        ) from None
    except OSError as e:
        raise ValueError(
            f"Téléchargement impossible depuis {url} : {e}.\n"
            f"Réseau coupé, ou serveur injoignable. Relancez : la reprise "
            f"repartira d'où elle s'est arrêtée."
        ) from None

    # Le serveur peut ignorer la demande de reprise : on repart alors de zéro.
    reprise_acceptee = getattr(reponse, "status", 200) == 206
    if deja and not reprise_acceptee:
        dire("  le serveur refuse la reprise, téléchargement complet.")
        deja = 0

    longueur = reponse.headers.get("Content-Length")
    total = (int(longueur) + deja) if longueur else None
    if total:
        dire(f"  {total / 2**30:.2f} Gio à récupérer")

    mode = "ab" if deja else "wb"
    recu = deja
    dernier_affichage = deja
    with reponse, partiel.open(mode) as fh:
        while True:
            morceau = reponse.read(1 << 20)  # 1 Mio
            if not morceau:
                break
            fh.write(morceau)
            recu += len(morceau)
            if recu - dernier_affichage >= 100 * 2**20:  # tous les 100 Mio
                dernier_affichage = recu
                if total:
                    dire(f"  {recu / 2**30:.2f} / {total / 2**30:.2f} Gio "
                         f"({recu * 100 / total:.0f} %)")
                else:
                    dire(f"  {recu / 2**30:.2f} Gio")

    if total and recu < total:
        raise ValueError(
            f"Téléchargement incomplet : {recu} octets sur {total} attendus. "
            f"Le fichier partiel est conservé, relancez pour reprendre."
        )

    partiel.replace(destination)
    dire(f"  terminé : {destination} ({recu / 2**30:.2f} Gio)")
    return destination


# --------------------------------------------------------------------------- #
# Lecture du dump
# --------------------------------------------------------------------------- #


def _ouvrir(chemin: Path):
    """Ouvre un dump, compressé ou non, en flux binaire."""
    if chemin.suffix == ".bz2":
        return bz2.open(chemin, "rb")
    if chemin.suffix == ".gz":
        return gzip.open(chemin, "rb")
    return chemin.open("rb")


@dataclass
class Article:
    titre: str
    texte: str


def lire_dump(chemin: str | Path, longueur_min: int = 200) -> Iterator[Article]:
    """Parcourt un dump XML de Wikipédia et rend les articles nettoyés.

    Ne conserve que l'espace de noms principal (`ns` valant 0) : on écarte ainsi
    les pages de discussion, les modèles, les catégories et l'aide, qui ne sont
    pas des articles. Les redirections sont également écartées — ce sont des
    pages d'une ligne qui pointent ailleurs.
    """
    from xml.etree import ElementTree

    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(
            f"Dump introuvable : {chemin}\n"
            f"Voir « futo data telecharger » pour la marche à suivre."
        )

    with _ouvrir(chemin) as flux:
        contexte = ElementTree.iterparse(flux, events=("end",))
        for _, element in contexte:
            # Les dumps portent un espace de noms XML : on compare donc le nom
            # local, jamais le nom complet.
            nom = element.tag.rsplit("}", 1)[-1]
            if nom != "page":
                continue

            try:
                titre = ""
                espace = "0"
                brut = None
                redirection = False
                for enfant in element:
                    nom_enfant = enfant.tag.rsplit("}", 1)[-1]
                    if nom_enfant == "title":
                        titre = (enfant.text or "").strip()
                    elif nom_enfant == "ns":
                        espace = (enfant.text or "0").strip()
                    elif nom_enfant == "redirect":
                        redirection = True
                    elif nom_enfant == "revision":
                        for petit in enfant:
                            if petit.tag.rsplit("}", 1)[-1] == "text":
                                brut = petit.text

                if espace == "0" and not redirection and brut:
                    texte = nettoyer_wikitexte(brut)
                    if len(texte) >= longueur_min:
                        yield Article(titre=titre, texte=texte)
            finally:
                # Indispensable : sans cette libération, l'arbre XML grandit
                # jusqu'à saturer la mémoire sur un dump de plusieurs gigaoctets.
                element.clear()


# --------------------------------------------------------------------------- #
# Conversion complète
# --------------------------------------------------------------------------- #


@dataclass
class ResultatConversion:
    n_articles: int
    n_caracteres: int
    chemin: Path

    def __str__(self) -> str:
        mo = self.n_caracteres / 1e6
        # Estimation à partir du rapport mesuré sur le corpus d'exemple.
        tokens = self.n_caracteres / 3.6
        return (
            f"{self.n_articles:,} articles · {mo:,.1f} Mo de texte · "
            f"environ {tokens / 1e6:,.0f} M tokens"
        ).replace(",", " ")


def convertir_dump(
    dump: str | Path,
    sortie: str | Path,
    articles_max: int | None = None,
    longueur_min: int = 200,
    journal=None,
) -> ResultatConversion:
    """Convertit un dump en JSONL prêt pour `futo data preparer`.

    Une ligne par article, avec un champ `text` — le format qu'attend le reste
    de la chaîne. Le titre est conservé dans un champ `titre`, ignoré à
    l'entraînement mais précieux pour retrouver un article suspect.
    """
    sortie = Path(sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)

    def dire(message: str) -> None:
        if journal is not None:
            journal(message)

    n_articles = 0
    n_caracteres = 0
    temporaire = sortie.with_suffix(sortie.suffix + ".partiel")

    try:
        with temporaire.open("w", encoding="utf-8") as fh:
            for article in lire_dump(dump, longueur_min=longueur_min):
                fh.write(
                    json.dumps(
                        {"text": article.texte, "titre": article.titre},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_articles += 1
                n_caracteres += len(article.texte)
                if n_articles % 20_000 == 0:
                    dire(
                        f"  {n_articles:,} articles · "
                        f"{n_caracteres / 1e6:,.0f} Mo".replace(",", " ")
                    )
                if articles_max is not None and n_articles >= articles_max:
                    dire(f"  arrêt volontaire à {articles_max} articles.")
                    break
        # Renommage final : un fichier tronqué par un plantage reste en
        # « .partiel » et ne peut pas être pris pour un corpus complet.
        temporaire.replace(sortie)
    except BaseException:
        temporaire.unlink(missing_ok=True)
        raise

    if n_articles == 0:
        raise ValueError(
            f"Aucun article extrait de {dump}. Est-ce bien un dump "
            f"« pages-articles » de Wikipédia ?"
        )

    return ResultatConversion(
        n_articles=n_articles, n_caracteres=n_caracteres, chemin=sortie
    )


def consigner_source(
    fichier_sources: str | Path,
    nom: str,
    dump: str | Path,
    resultat: ResultatConversion,
    licence: str = "CC BY-SA 4.0",
) -> None:
    """Ajoute une ligne à `data/SOURCES.md`.

    Ce n'est pas de la bureaucratie : le règlement européen sur l'IA impose de
    publier un résumé détaillé des données d'entraînement, et cette liste est
    impossible à reconstituer six mois plus tard. Cinq secondes maintenant,
    une journée perdue plus tard.
    """
    chemin = Path(fichier_sources)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    nouveau = not chemin.exists()

    with chemin.open("a", encoding="utf-8") as fh:
        if nouveau:
            fh.write(
                "# Sources des données d'entraînement\n\n"
                "Tenu à jour à chaque ajout de corpus. Exigé par le règlement\n"
                "européen sur l'IA, et impossible à reconstituer après coup.\n\n"
                "| Date | Source | Fichier | Articles | Taille | Licence |\n"
                "|---|---|---|---|---|---|\n"
            )
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        taille = f"{resultat.n_caracteres / 1e6:.0f} Mo"
        articles = f"{resultat.n_articles:,}".replace(",", " ")
        fh.write(
            f"| {date} | {nom} | `{Path(dump).name}` | {articles} | "
            f"{taille} | {licence} |\n"
        )
