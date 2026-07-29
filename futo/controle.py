"""Contrôle de la qualité d'un corpus avant de l'entraîner.

Une chaîne de nettoyage qui laisse passer 2 % de balisage ne se voit pas à
l'œil : on regarde trois articles, ils sont propres, on lance la conversion
complète, et on entraîne un modèle sur du texte pollué. Le défaut ne se révèle
qu'à la génération, des jours plus tard.

Ce module répond à la question « puis-je entraîner là-dessus ? » en quelques
secondes, en comptant ce qui n'aurait pas dû survivre et en montrant où. Il est
conçu pour être lancé sur un ÉCHANTILLON avant d'engager la conversion entière.

Trois choses sont mesurées :

1. **Les traces de balisage** — accolades de modèle, crochets de lien, balises,
   barres de tableau. Zéro est l'objectif ; une trace pour mille documents est
   tolérable ; quelques pourcents veulent dire que le nettoyage est à revoir.
2. **La typographie française** — accents, ligatures, apostrophes, guillemets.
   Un corpus français qui n'en contient presque pas signale un problème
   d'encodage ou une langue mal détectée, bien avant l'entraînement.
3. **La forme des documents** — longueurs, doublons exacts. Un corpus plein de
   quasi-doublons se mémorise au lieu de s'apprendre.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Controle", "controler_corpus", "TRACES_DE_BALISAGE"]


# Ce qui ne devrait jamais survivre à un nettoyage correct. Chaque motif est
# accompagné de ce qu'il révèle, pour que le rapport soit actionnable plutôt
# que décoratif.
TRACES_DE_BALISAGE: list[tuple[str, str, str]] = [
    ("modele_ouvrant", r"\{\{", "modèle non supprimé — imbrication mal gérée"),
    ("modele_fermant", r"\}\}", "fermeture de modèle orpheline"),
    ("lien_ouvrant", r"\[\[", "lien interne non traité"),
    ("lien_fermant", r"\]\]", "fermeture de lien orpheline"),
    ("tableau", r"\{\||\|\}", "tableau non supprimé"),
    ("balise_ref", r"<ref[\s/>]", "référence non supprimée"),
    ("balise_html", r"</?(?:div|span|table|tr|td|br|small|sup|sub)\b", "HTML résiduel"),
    ("entite_html", r"&(?:nbsp|amp|lt|gt|quot|#\d+);", "entité HTML non décodée"),
    ("gras_italique", r"'{3,}", "balisage de gras non retiré"),
    ("titre_wiki", r"^={2,}.*={2,}$", "titre de section non converti"),
    ("url_nue", r"https?://\S{10,}", "adresse laissée telle quelle"),
]

# Marqueurs du français, pour repérer un corpus qui n'en serait pas.
_ACCENTS = "àâäéèêëîïôöùûüÿçÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ"
_LIGATURES = "œŒæÆ"
# Pour la détection de langue, les ligatures comptent autant que les accents :
# « cœur », « sœur », « œuvre » et « bœuf » sont parmi les mots les plus
# français qui soient, et n'auraient été comptés nulle part sans cette union.
_MARQUEURS_FR = _ACCENTS + _LIGATURES
_MOTIF_MOT = re.compile(r"\w+", re.UNICODE)


def _milliers(n: float) -> str:
    """Formate un entier à la française : 1 234 567.

    À appliquer nombre par nombre. Formater une phrase entière puis y
    remplacer les virgules avale aussi celles de la ponctuation — erreur
    déjà commise deux fois dans ce projet.
    """
    return f"{n:,.0f}".replace(",", "\u202f")


@dataclass
class Controle:
    """Rapport de contrôle. Lisible tel quel dans un terminal."""

    n_documents: int
    n_octets: int
    n_mots: int
    traces: dict[str, int] = field(default_factory=dict)
    exemples: dict[str, str] = field(default_factory=dict)
    docs_touches: dict[str, int] = field(default_factory=dict)
    typographie: dict[str, int] = field(default_factory=dict)
    longueurs: list[int] = field(default_factory=list)
    n_doublons: int = 0
    n_vides: int = 0

    @property
    def tokens_estimes(self) -> float:
        """Estimation à 3,56 octets par token — rapport mesuré sur Wikipédia FR."""
        return self.n_octets / 3.56

    @property
    def part_touchee(self) -> float:
        """Part des documents portant au moins une trace de balisage."""
        if not self.n_documents:
            return 0.0
        return max(self.docs_touches.values(), default=0) / self.n_documents

    def verdict(self) -> tuple[bool, str]:
        """Peut-on entraîner là-dessus ?

        Les seuils sont volontairement sévères sur le balisage : c'est du bruit
        que le modèle apprendrait à reproduire.
        """
        part = self.part_touchee
        if part > 0.05:
            return False, (
                f"{part * 100:.1f} % des documents portent une trace de balisage. "
                f"Le nettoyage est à revoir avant d'entraîner."
            )
        if part > 0.005:
            return True, (
                f"{part * 100:.2f} % des documents portent une trace de balisage. "
                f"Acceptable, mais regardez les exemples ci-dessus."
            )
        accents = self.typographie.get("mots_accentues", 0)
        if self.n_mots and accents / self.n_mots < 0.05:
            return False, (
                f"Seulement {accents / max(self.n_mots, 1) * 100:.1f} % de mots "
                f"accentués : ce corpus est-il vraiment en français ?"
            )
        return True, "Corpus propre — rien ne s'oppose à l'entraînement."

    def __str__(self) -> str:
        tokens = self.tokens_estimes
        volume = (
            f"{self.n_octets / 1e9:.2f} Go" if self.n_octets >= 1e9
            else f"{self.n_octets / 1e6:.1f} Mo"
        )
        en_tokens = (
            f"{tokens / 1e9:.2f} G tokens" if tokens >= 1e9
            else f"{tokens / 1e6:.2f} M tokens"
        )
        lignes = [
            f"{_milliers(self.n_documents)} documents · {volume} · "
            f"{_milliers(self.n_mots)} mots · ~{en_tokens}",
        ]
        if self.longueurs:
            tries = sorted(self.longueurs)
            n = len(tries)
            lignes.append(
                f"  longueur des documents : médiane {_milliers(tries[n // 2])} car., "
                f"1er décile {_milliers(tries[n // 10])}, "
                f"9e décile {_milliers(tries[9 * n // 10])}"
            )
        if self.n_vides:
            lignes.append(f"  {_milliers(self.n_vides)} document(s) vide(s)")
        if self.n_doublons:
            part = self.n_doublons * 100 / max(self.n_documents, 1)
            lignes.append(
                f"  {_milliers(self.n_doublons)} doublon(s) exact(s) ({part:.1f} %)"
            )

        lignes.append("")
        lignes.append("Traces de balisage (zéro est l'objectif) :")
        restes = {c: n for c, n in self.traces.items() if n}
        if not restes:
            lignes.append("  aucune.")
        else:
            for cle, nombre in sorted(restes.items(), key=lambda kv: -kv[1]):
                description = next(d for c, _, d in TRACES_DE_BALISAGE if c == cle)
                docs = self.docs_touches.get(cle, 0)
                lignes.append(
                    f"  {cle:<16} {_milliers(nombre):>9} occurrence(s) dans "
                    f"{_milliers(docs)} document(s) — {description}"
                )
                if cle in self.exemples:
                    lignes.append(f"      … {self.exemples[cle]} …")

        lignes.append("")
        lignes.append("Typographie française :")
        for cle, libelle in (
            ("mots_accentues", "mots accentués ou ligaturés"),
            ("ligatures", "ligatures œ / æ"),
            ("apostrophes_droites", "apostrophes droites"),
            ("apostrophes_typo", "apostrophes typographiques"),
            ("guillemets", "guillemets « »"),
            ("insecables", "espaces insécables"),
        ):
            valeur = self.typographie.get(cle, 0)
            ligne = f"  {libelle:<28} {_milliers(valeur):>11}"
            if cle == "mots_accentues" and self.n_mots:
                ligne += f"  ({valeur * 100 / self.n_mots:.1f} % des mots)"
            lignes.append(ligne)

        ok, message = self.verdict()
        lignes.append("")
        lignes.append(("✓ " if ok else "✗ ") + message)
        return "\n".join(lignes)


def _documents(chemin: Path):
    """Rend les documents d'un fichier JSONL ou texte."""
    if chemin.name.endswith((".jsonl", ".jsonl.gz")):
        from .data import _ouvrir_texte

        with _ouvrir_texte(chemin) as fh:
            for ligne in fh:
                ligne = ligne.strip()
                if not ligne:
                    continue
                obj = json.loads(ligne)
                for cle in ("text", "texte", "content", "contenu"):
                    if isinstance(obj.get(cle), str):
                        yield obj[cle]
                        break
    else:
        from .data import lire_documents

        yield from lire_documents(chemin, "ligne-vide")


def controler_corpus(
    chemins: list[str | Path],
    documents_max: int | None = None,
    longueur_exemple: int = 90,
) -> Controle:
    """Analyse un corpus et rend un rapport lisible.

    `documents_max` permet de contrôler un échantillon d'un très gros fichier
    sans le parcourir en entier — c'est le mode d'emploi normal avant une
    conversion complète.
    """
    motifs = [
        (cle, re.compile(expression, re.MULTILINE)) for cle, expression, _ in TRACES_DE_BALISAGE
    ]
    traces: Counter[str] = Counter()
    docs_touches: Counter[str] = Counter()
    exemples: dict[str, str] = {}
    typo: Counter[str] = Counter()
    longueurs: list[int] = []
    empreintes: set[str] = set()

    n_documents = n_octets = n_mots = n_doublons = n_vides = 0

    for chemin in chemins:
        chemin = Path(chemin)
        if not chemin.exists():
            raise FileNotFoundError(f"Corpus introuvable : {chemin}")

        for texte in _documents(chemin):
            n_documents += 1
            if not texte.strip():
                n_vides += 1
                continue

            n_octets += len(texte.encode("utf-8"))
            longueurs.append(len(texte))

            empreinte = hashlib.blake2b(texte.encode("utf-8"), digest_size=16).hexdigest()
            if empreinte in empreintes:
                n_doublons += 1
            else:
                empreintes.add(empreinte)

            mots = _MOTIF_MOT.findall(texte)
            n_mots += len(mots)
            typo["mots_accentues"] += sum(
                1 for m in mots if any(c in _MARQUEURS_FR for c in m)
            )
            typo["ligatures"] += sum(texte.count(c) for c in _LIGATURES)
            typo["apostrophes_droites"] += texte.count("'")
            typo["apostrophes_typo"] += texte.count("’")
            typo["guillemets"] += texte.count("«") + texte.count("»")
            typo["insecables"] += texte.count(" ") + texte.count(" ")

            for cle, motif in motifs:
                trouves = motif.findall(texte)
                if not trouves:
                    continue
                traces[cle] += len(trouves)
                docs_touches[cle] += 1
                if cle not in exemples:
                    position = motif.search(texte).start()
                    debut = max(0, position - longueur_exemple // 2)
                    extrait = texte[debut : debut + longueur_exemple]
                    exemples[cle] = extrait.replace("\n", " ⏎ ")

            if documents_max is not None and n_documents >= documents_max:
                break
        if documents_max is not None and n_documents >= documents_max:
            break

    return Controle(
        n_documents=n_documents,
        n_octets=n_octets,
        n_mots=n_mots,
        traces=dict(traces),
        exemples=exemples,
        docs_touches=dict(docs_touches),
        typographie=dict(typo),
        longueurs=longueurs,
        n_doublons=n_doublons,
        n_vides=n_vides,
    )


def extraire_echantillon(
    chemins: list[str | Path], n: int = 3, longueur: int = 700
) -> list[str]:
    """Rend quelques documents entiers, pour les lire à l'œil.

    Les compteurs disent qu'un corpus est propre ; seule la lecture dit qu'il
    est *lisible*. Les deux sont nécessaires.
    """
    echantillon: list[str] = []
    for chemin in chemins:
        for texte in _documents(Path(chemin)):
            texte = unicodedata.normalize("NFC", texte).strip()
            if len(texte) < 200:
                continue
            echantillon.append(texte[:longueur])
            if len(echantillon) >= n:
                return echantillon
    return echantillon
