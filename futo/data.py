"""Préparation et lecture des données d'entraînement.

Format des shards
-----------------
Un shard est un fichier binaire auto-descriptif : un en-tête de 1 024 octets
(256 entiers 32 bits, petit-boutiste) suivi des tokens bruts.

    position  taille  contenu
    0         4       nombre magique 0x4655544F (« FUTO » en ASCII)
    4         4       version du format (1)
    8         4       code du type : 1 = uint16, 2 = uint32
    12        4       nombre de tokens dans le fichier
    16        4       taille du vocabulaire ayant servi à l'encodage
    20        1004    réservé (zéros)
    1024      …       les tokens, à la suite

Pourquoi un en-tête plutôt qu'un fichier annexe : un shard reste
compréhensible tout seul, on ne peut pas se retrouver avec un `.bin` orphelin
dont on ignore le type. Pourquoi 1 024 octets : c'est un multiple de 2 et de 4,
donc les tokens restent alignés en mémoire, ce qui compte pour `mmap`.

Le champ `vocab_size` sert de garde-fou : réutiliser des shards préparés avec un
autre tokenizer produirait un modèle silencieusement incohérent. On vérifie donc
à l'ouverture.

Tirage des lots
---------------
Le chargeur ne garde aucun état : le lot du pas *n* est entièrement déterminé
par `(graine, n)`. Deux conséquences utiles :

* reprendre un entraînement au pas 4 200 redonne *exactement* la même suite de
  lots que si l'on n'avait jamais interrompu — aucun état de chargeur à
  sérialiser, donc rien qui puisse se désynchroniser ;
* en multi-GPU, chaque rang calcule sa part du lot sans communication.

Le prix à payer est qu'on tire les fenêtres au hasard avec remise, au lieu de
parcourir une permutation exacte du corpus. Sur un pré-entraînement d'une seule
époque, la différence est négligeable.
"""

from __future__ import annotations

import gzip
import json
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "MAGIQUE",
    "VERSION_FORMAT",
    "EnteteShard",
    "ecrire_shard",
    "ouvrir_shard",
    "preparer_corpus",
    "ChargeurTokens",
]

MAGIQUE = 0x4655544F
VERSION_FORMAT = 1


def _milliers(n: int) -> str:
    """Formate un entier à la française : 1 234 567, avec des espaces fines."""
    return f"{n:,}".replace(",", " ")


TAILLE_ENTETE = 1_024  # octets
_CODES_DTYPE = {1: np.uint16, 2: np.uint32}
_DTYPE_VERS_CODE = {np.dtype(np.uint16): 1, np.dtype(np.uint32): 2}


# --------------------------------------------------------------------------- #
# Lecture et écriture des shards
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EnteteShard:
    version: int
    dtype: np.dtype
    n_tokens: int
    vocab_size: int


def ecrire_shard(
    chemin: str | Path, tokens: np.ndarray, vocab_size: int
) -> EnteteShard:
    """Écrit un tableau de tokens dans un shard."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    dtype = np.dtype(tokens.dtype)
    if dtype not in _DTYPE_VERS_CODE:
        raise ValueError(f"Type non pris en charge : {dtype} (attendu uint16 ou uint32).")
    if tokens.size and int(tokens.max()) >= vocab_size:
        raise ValueError(
            f"Un token vaut {int(tokens.max())}, hors du vocabulaire de {vocab_size}."
        )

    entete = np.zeros(256, dtype=np.int32)
    entete[0] = MAGIQUE
    entete[1] = VERSION_FORMAT
    entete[2] = _DTYPE_VERS_CODE[dtype]
    entete[3] = tokens.size
    entete[4] = vocab_size

    with chemin.open("wb") as fh:
        fh.write(entete.astype("<i4").tobytes())
        fh.write(np.ascontiguousarray(tokens).astype(dtype.newbyteorder("<")).tobytes())

    return EnteteShard(VERSION_FORMAT, dtype, int(tokens.size), vocab_size)


def lire_entete(chemin: str | Path) -> EnteteShard:
    """Lit le seul en-tête, sans toucher aux tokens."""
    chemin = Path(chemin)
    with chemin.open("rb") as fh:
        brut = fh.read(TAILLE_ENTETE)
    if len(brut) < TAILLE_ENTETE:
        raise ValueError(f"{chemin} : fichier tronqué, en-tête incomplet.")
    entete = np.frombuffer(brut, dtype="<i4", count=256)

    if int(entete[0]) != MAGIQUE:
        raise ValueError(
            f"{chemin} : ce n'est pas un shard Futo "
            f"(nombre magique {int(entete[0]):#x} au lieu de {MAGIQUE:#x})."
        )
    version = int(entete[1])
    if version != VERSION_FORMAT:
        raise ValueError(
            f"{chemin} : format version {version}, or ce code lit la version "
            f"{VERSION_FORMAT}. Reconstruisez les shards avec « futo data preparer »."
        )
    code = int(entete[2])
    if code not in _CODES_DTYPE:
        raise ValueError(f"{chemin} : code de type inconnu ({code}).")

    return EnteteShard(
        version=version,
        dtype=np.dtype(_CODES_DTYPE[code]),
        n_tokens=int(entete[3]),
        vocab_size=int(entete[4]),
    )


def ouvrir_shard(chemin: str | Path) -> tuple[np.memmap, EnteteShard]:
    """Projette un shard en mémoire virtuelle, sans le charger.

    `mmap` laisse le système d'exploitation gérer le cache : on peut lire au
    hasard dans 200 Go de tokens avec 16 Go de RAM.
    """
    chemin = Path(chemin)
    entete = lire_entete(chemin)
    tableau = np.memmap(
        chemin,
        dtype=entete.dtype,
        mode="r",
        offset=TAILLE_ENTETE,
        shape=(entete.n_tokens,),
    )
    return tableau, entete


# --------------------------------------------------------------------------- #
# Préparation du corpus
# --------------------------------------------------------------------------- #


def _ouvrir_texte(chemin: Path):
    if chemin.suffix == ".gz":
        return gzip.open(chemin, "rt", encoding="utf-8", errors="replace")
    return chemin.open("r", encoding="utf-8", errors="replace")


def lire_documents(chemin: Path, separateur: str = "ligne-vide") -> Iterator[str]:
    """Découpe un fichier en documents.

    Trois conventions :

    * ``jsonl`` — un objet JSON par ligne, dont on prend le champ ``text``
      (ou ``texte``, ou ``content``). C'est le format des grands corpus publics ;
    * ``ligne-vide`` — les paragraphes séparés par une ligne blanche, adapté aux
      fichiers texte ordinaires ;
    * ``fichier`` — le fichier entier est un seul document.
    """
    if separateur == "jsonl" or chemin.name.endswith((".jsonl", ".jsonl.gz")):
        with _ouvrir_texte(chemin) as fh:
            for numero, ligne in enumerate(fh, 1):
                ligne = ligne.strip()
                if not ligne:
                    continue
                try:
                    obj = json.loads(ligne)
                except json.JSONDecodeError:
                    raise ValueError(f"{chemin}:{numero} : ligne JSON invalide.") from None
                for cle in ("text", "texte", "content", "contenu"):
                    if cle in obj and isinstance(obj[cle], str):
                        if obj[cle].strip():
                            yield obj[cle]
                        break
                else:
                    raise ValueError(
                        f"{chemin}:{numero} : aucun champ de texte "
                        f"(cherchés : text, texte, content, contenu)."
                    )
        return

    with _ouvrir_texte(chemin) as fh:
        contenu = fh.read()

    if separateur == "fichier":
        if contenu.strip():
            yield contenu
        return

    if separateur != "ligne-vide":
        raise ValueError(
            f"Séparateur inconnu : {separateur!r} (attendu : ligne-vide, fichier ou jsonl)."
        )
    for bloc in contenu.split("\n\n"):
        if bloc.strip():
            yield bloc.strip() + "\n"


@dataclass
class ResultatPreparation:
    n_documents: int
    n_tokens: int
    n_shards_train: int
    n_shards_val: int
    n_tokens_val: int
    octets_source: int

    def __str__(self) -> str:
        octets_par_token = self.octets_source / max(self.n_tokens, 1)
        return (
            f"{_milliers(self.n_documents)} documents · "
            f"{_milliers(self.n_tokens)} tokens "
            f"({self.n_shards_train} shard(s) d'entraînement, "
            f"{_milliers(self.n_tokens_val)} tokens de validation) · "
            f"{octets_par_token:.2f} octet/token"
        )


def preparer_corpus(
    fichiers: list[str | Path],
    tokenizer,
    dossier_sortie: str | Path,
    tokens_par_shard: int = 100_000_000,
    fraction_val: float = 0.005,
    tokens_val_max: int = 10_000_000,
    separateur: str = "ligne-vide",
    prefixe_train: str = "train",
    prefixe_val: str = "val",
    normaliser: bool = True,
    taille_lot: int = 1_000,
    journal=None,
) -> ResultatPreparation:
    """Encode un corpus texte en shards binaires.

    Les documents sont encodés puis concaténés, chacun suivi du token
    `<|fin_de_texte|>`. Le modèle apprend ainsi où un document s'arrête, ce qui
    lui permet de ne pas enchaîner deux textes sans rapport.

    La validation est prélevée en tête du flux, avant tout mélange : elle reste
    donc identique d'une préparation à l'autre tant que le corpus ne change pas,
    ce qui rend les perplexités comparables entre expériences.
    """
    chemins = [Path(f) for f in fichiers]
    manquants = [str(c) for c in chemins if not c.exists()]
    if manquants:
        raise FileNotFoundError(f"Corpus introuvable : {manquants}")

    dossier_sortie = Path(dossier_sortie)
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    dtype = tokenizer.dtype_shards
    vocab_size = tokenizer.vocab_size

    def dire(message: str) -> None:
        if journal is not None:
            journal(message)

    octets_source = sum(c.stat().st_size for c in chemins)
    # Estimation prudente : ~3 octets par token en français avec ce tokenizer.
    # Elle ne sert qu'à dimensionner la part de validation.
    tokens_estimes = max(int(octets_source / 3), 1)
    objectif_val = min(int(tokens_estimes * fraction_val), tokens_val_max)

    tampon: list[int] = []
    tampon_val: list[int] = []
    n_documents = 0
    n_tokens = 0
    n_shards_train = 0
    n_shards_val = 0

    def vider(tampon_local: list[int], prefixe: str, indice: int) -> int:
        chemin = dossier_sortie / f"{prefixe}_{indice:05d}.bin"
        ecrire_shard(chemin, np.asarray(tampon_local, dtype=dtype), vocab_size)
        dire(f"  écrit {chemin.name} ({_milliers(len(tampon_local))} tokens)")
        return indice + 1

    lot_textes: list[str] = []

    def encoder_lot() -> None:
        nonlocal n_documents, n_tokens, n_shards_train, n_shards_val
        if not lot_textes:
            return
        for ids in tokenizer.encoder_lot(lot_textes, ajouter_fin=True):
            n_documents += 1
            n_tokens += len(ids)
            if len(tampon_val) < objectif_val:
                tampon_val.extend(ids)
            else:
                tampon.extend(ids)
                while len(tampon) >= tokens_par_shard:
                    n_shards_train = vider(tampon[:tokens_par_shard], prefixe_train, n_shards_train)
                    del tampon[:tokens_par_shard]
        lot_textes.clear()

    for chemin in chemins:
        dire(f"lecture de {chemin}")
        for document in lire_documents(chemin, separateur):
            if normaliser:
                # NFC avant tout : la même règle qu'à l'entraînement du
                # tokenizer, sans quoi « é » précomposé et « é » décomposé
                # produiraient deux suites de tokens différentes.
                document = unicodedata.normalize("NFC", document)
            lot_textes.append(document)
            if len(lot_textes) >= taille_lot:
                encoder_lot()
    encoder_lot()

    if tampon_val:
        n_shards_val = vider(tampon_val, prefixe_val, n_shards_val)
    if tampon:
        n_shards_train = vider(tampon, prefixe_train, n_shards_train)

    if n_shards_train == 0:
        raise ValueError(
            "Aucun shard d'entraînement produit : le corpus est-il vide, ou "
            "entièrement absorbé par la validation ? Réduisez fraction_val."
        )

    return ResultatPreparation(
        n_documents=n_documents,
        n_tokens=n_tokens,
        n_shards_train=n_shards_train,
        n_shards_val=n_shards_val,
        n_tokens_val=len(tampon_val),
        octets_source=octets_source,
    )


# --------------------------------------------------------------------------- #
# Chargement pendant l'entraînement
# --------------------------------------------------------------------------- #


class ChargeurTokens:
    """Fournit les lots (entrée, cible) en tirant des fenêtres dans les shards.

    Sans état : `lot(pas)` renvoie toujours le même lot pour un même pas et une
    même graine. C'est ce qui rend la reprise après plantage exacte.
    """

    def __init__(
        self,
        dossier: str | Path,
        prefixe: str,
        block_size: int,
        batch_size: int,
        seed: int = 1234,
        rang: int = 0,
        monde: int = 1,
    ) -> None:
        dossier = Path(dossier)
        chemins = sorted(dossier.glob(f"{prefixe}_*.bin"))
        if not chemins:
            raise FileNotFoundError(
                f"Aucun shard « {prefixe}_*.bin » dans {dossier}.\n"
                f"Préparez les données avec : futo data preparer"
            )

        self.shards: list[np.memmap] = []
        self.entetes: list[EnteteShard] = []
        for chemin in chemins:
            tableau, entete = ouvrir_shard(chemin)
            if entete.n_tokens <= block_size:
                # Un shard plus court qu'une fenêtre ne peut fournir aucun
                # exemple : on le signale plutôt que de le sauter en silence.
                raise ValueError(
                    f"{chemin.name} ne contient que {entete.n_tokens} tokens, "
                    f"moins que le contexte demandé ({block_size} + 1)."
                )
            self.shards.append(tableau)
            self.entetes.append(entete)

        vocabs = {e.vocab_size for e in self.entetes}
        if len(vocabs) > 1:
            raise ValueError(
                f"Shards incohérents dans {dossier} : plusieurs tailles de "
                f"vocabulaire {sorted(vocabs)}. Ils n'ont pas été préparés avec "
                f"le même tokenizer."
            )
        self.vocab_size = next(iter(vocabs))

        self.chemins = chemins
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = seed
        self.rang = rang
        self.monde = monde

        # Le nombre de fenêtres de départ possibles par shard.
        self.departs = np.array(
            [e.n_tokens - block_size - 1 for e in self.entetes], dtype=np.int64
        )
        self.total_departs = int(self.departs.sum())
        self.cumul = np.cumsum(self.departs)
        self.n_tokens = int(sum(e.n_tokens for e in self.entetes))

    def __len__(self) -> int:
        return self.n_tokens

    def lot(self, pas: int):
        """Renvoie (entrée, cible), deux tenseurs int64 de forme (batch_size, block_size)."""
        import torch

        # Une graine par (graine globale, pas, rang) : reproductible, et chaque
        # rang d'un entraînement distribué voit des fenêtres différentes.
        rng = np.random.default_rng((self.seed, pas, self.rang))
        tirages = rng.integers(0, self.total_departs, size=self.batch_size)

        entrees = np.empty((self.batch_size, self.block_size), dtype=np.int64)
        cibles = np.empty((self.batch_size, self.block_size), dtype=np.int64)
        for i, tirage in enumerate(tirages):
            indice_shard = int(np.searchsorted(self.cumul, tirage, side="right"))
            debut_shard = 0 if indice_shard == 0 else int(self.cumul[indice_shard - 1])
            depart = int(tirage - debut_shard)
            fenetre = self.shards[indice_shard][depart : depart + self.block_size + 1]
            # Copie explicite : la memmap peut être invalidée, et astype force
            # la sortie de uint16 (que PyTorch ne connaît pas) vers int64.
            fenetre = np.asarray(fenetre, dtype=np.int64)
            entrees[i] = fenetre[:-1]
            cibles[i] = fenetre[1:]

        return torch.from_numpy(entrees), torch.from_numpy(cibles)
