"""Export au format GGUF, celui que lisent llama.cpp et ses dérivés.

Pourquoi ce format
------------------
GGUF est un fichier unique qui porte tout : les métadonnées de l'architecture,
le vocabulaire du tokenizer, et les poids. C'est ce qu'attendent llama.cpp et
les applications construites dessus — dont les claviers à suggestions qui
embarquent un modèle de langue local.

Notre modèle se convertit sans acrobatie parce qu'il a exactement la forme de
l'architecture `llama` : RMSNorm en pré-norme, RoPE, attention à requêtes
groupées, SwiGLU, aucun biais, embeddings liés. Chaque tenseur a son
correspondant, un pour un.

Ce que ce module fait, et ne fait pas
-------------------------------------
Il écrit un GGUF en float16 ou float32. Il ne quantifie pas : la quantification
en q4, q6 ou q8 est le métier de `llama-quantize`, livré avec llama.cpp, et le
refaire ici serait une mauvaise idée — les schémas k-quants sont subtils et
évoluent.

    futo exporter gguf sorties/mac/dernier.pt --sortie futo-mac-f16.gguf
    llama-quantize futo-mac-f16.gguf futo-mac-q6_k.gguf Q6_K

RÉSERVE IMPORTANTE SUR LE TOKENIZER
------------------------------------
Le vocabulaire et les fusions sont écrits dans le fichier, et ils sont exacts.
Mais llama.cpp ne rejoue pas notre découpe préalable : il applique l'une des
expressions régulières de sa propre table, désignée par `tokenizer.ggml.pre`.
Notre découpe française — élisions soudées au mot qui les porte, clitiques
attachés — n'y figure pas.

Conséquence concrète : un texte donné à llama.cpp ne sera pas découpé
exactement comme il l'a été à l'entraînement. Le modèle fonctionnera, mais en
dessous de ce qu'il vaut. Deux issues, aucune n'est un raccourci : faire
enregistrer notre découpe dans llama.cpp, ou fournir les identifiants de tokens
déjà encodés par notre tokenizer plutôt que du texte brut.

Cette réserve est écrite ici parce que c'est le genre de détail qui se paie
cher quand on le découvre après coup.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

__all__ = [
    "TYPES_VALEUR",
    "ecrire_gguf",
    "lire_gguf",
    "metadonnees_depuis_config",
    "tenseurs_depuis_etat",
    "exporter_gguf",
]

MAGIE = b"GGUF"
VERSION = 3
ALIGNEMENT = 32

# Types de valeur des métadonnées, tels que définis par la spécification GGUF.
TYPES_VALEUR = {
    "uint8": 0, "int8": 1, "uint16": 2, "int16": 3,
    "uint32": 4, "int32": 5, "float32": 6, "bool": 7,
    "string": 8, "array": 9, "uint64": 10, "int64": 11, "float64": 12,
}
# Types de tenseur GGML. Seuls les formats non quantifiés nous concernent :
# la quantification est faite ensuite par llama-quantize.
TYPE_F32, TYPE_F16 = 0, 1


# --------------------------------------------------------------------------- #
# Écriture
# --------------------------------------------------------------------------- #


def _ecrire_chaine(sortie, texte: str) -> None:
    brut = texte.encode("utf-8")
    sortie.write(struct.pack("<Q", len(brut)))
    sortie.write(brut)


def _ecrire_valeur(sortie, valeur: Any) -> None:
    """Écrit une valeur typée. Le type est déduit de la valeur Python.

    Les entiers sont écrits en uint32 tant qu'ils y tiennent : c'est ce
    qu'attendent les lecteurs pour les champs d'architecture.
    """
    if isinstance(valeur, bool):
        sortie.write(struct.pack("<I", TYPES_VALEUR["bool"]))
        sortie.write(struct.pack("<?", valeur))
    elif isinstance(valeur, int):
        if 0 <= valeur < 2**32:
            sortie.write(struct.pack("<I", TYPES_VALEUR["uint32"]))
            sortie.write(struct.pack("<I", valeur))
        else:
            sortie.write(struct.pack("<I", TYPES_VALEUR["int64"]))
            sortie.write(struct.pack("<q", valeur))
    elif isinstance(valeur, float):
        sortie.write(struct.pack("<I", TYPES_VALEUR["float32"]))
        sortie.write(struct.pack("<f", valeur))
    elif isinstance(valeur, str):
        sortie.write(struct.pack("<I", TYPES_VALEUR["string"]))
        _ecrire_chaine(sortie, valeur)
    elif isinstance(valeur, (list, tuple)):
        sortie.write(struct.pack("<I", TYPES_VALEUR["array"]))
        _ecrire_tableau(sortie, list(valeur))
    else:
        raise TypeError(f"Type non exportable en GGUF : {type(valeur).__name__}")


def _ecrire_tableau(sortie, valeurs: list) -> None:
    if not valeurs:
        # Un tableau vide doit quand même annoncer un type : chaîne par défaut,
        # le seul cas rencontré en pratique (vocabulaire ou fusions absents).
        sortie.write(struct.pack("<I", TYPES_VALEUR["string"]))
        sortie.write(struct.pack("<Q", 0))
        return
    premier = valeurs[0]
    if isinstance(premier, str):
        sortie.write(struct.pack("<I", TYPES_VALEUR["string"]))
        sortie.write(struct.pack("<Q", len(valeurs)))
        for v in valeurs:
            _ecrire_chaine(sortie, v)
    elif isinstance(premier, bool):
        sortie.write(struct.pack("<I", TYPES_VALEUR["bool"]))
        sortie.write(struct.pack("<Q", len(valeurs)))
        for v in valeurs:
            sortie.write(struct.pack("<?", v))
    elif isinstance(premier, int):
        sortie.write(struct.pack("<I", TYPES_VALEUR["int32"]))
        sortie.write(struct.pack("<Q", len(valeurs)))
        for v in valeurs:
            sortie.write(struct.pack("<i", v))
    elif isinstance(premier, float):
        sortie.write(struct.pack("<I", TYPES_VALEUR["float32"]))
        sortie.write(struct.pack("<Q", len(valeurs)))
        for v in valeurs:
            sortie.write(struct.pack("<f", v))
    else:
        raise TypeError(f"Tableau non exportable : {type(premier).__name__}")


def ecrire_gguf(chemin: str | Path, metadonnees: dict, tenseurs: dict) -> Path:
    """Écrit un fichier GGUF complet.

    `tenseurs` associe un nom à un tableau numpy. Les dimensions sont inscrites
    en ordre INVERSE, la convention GGUF plaçant la dimension qui varie le plus
    vite en premier — une matrice PyTorch (sortie, entrée) devient donc
    (entrée, sortie) dans le fichier, sans que les octets bougent.
    """
    import numpy as np

    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    prepares = {}
    for nom, tableau in tenseurs.items():
        tableau = np.ascontiguousarray(tableau)
        if tableau.dtype == np.float16:
            code = TYPE_F16
        elif tableau.dtype == np.float32:
            code = TYPE_F32
        else:
            raise TypeError(
                f"{nom} : seuls float16 et float32 sont exportés, reçu {tableau.dtype}. "
                f"La quantification est le métier de llama-quantize."
            )
        prepares[nom] = (tableau, code)

    with open(chemin, "wb") as sortie:
        sortie.write(MAGIE)
        sortie.write(struct.pack("<I", VERSION))
        sortie.write(struct.pack("<Q", len(prepares)))
        sortie.write(struct.pack("<Q", len(metadonnees) + 1))

        # L'alignement doit figurer dans les métadonnées : le lecteur en a
        # besoin pour trouver le début des données.
        _ecrire_chaine(sortie, "general.alignment")
        _ecrire_valeur(sortie, ALIGNEMENT)
        for cle, valeur in metadonnees.items():
            _ecrire_chaine(sortie, cle)
            _ecrire_valeur(sortie, valeur)

        # Les décalages sont relatifs au début de la zone de données, qui
        # commence après un remplissage : on les calcule donc d'abord.
        decalage = 0
        decalages = {}
        for nom, (tableau, _) in prepares.items():
            decalages[nom] = decalage
            taille = tableau.nbytes
            decalage += taille + (-taille % ALIGNEMENT)

        for nom, (tableau, code) in prepares.items():
            _ecrire_chaine(sortie, nom)
            sortie.write(struct.pack("<I", tableau.ndim))
            for dimension in reversed(tableau.shape):
                sortie.write(struct.pack("<Q", int(dimension)))
            sortie.write(struct.pack("<I", code))
            sortie.write(struct.pack("<Q", decalages[nom]))

        position = sortie.tell()
        sortie.write(b"\0" * (-position % ALIGNEMENT))
        for tableau, _ in prepares.values():
            sortie.write(tableau.tobytes())
            sortie.write(b"\0" * (-tableau.nbytes % ALIGNEMENT))

    return chemin


# --------------------------------------------------------------------------- #
# Lecture — sert aux tests, et à vérifier un fichier produit ailleurs
# --------------------------------------------------------------------------- #


def _lire_chaine(source) -> str:
    (longueur,) = struct.unpack("<Q", source.read(8))
    return source.read(longueur).decode("utf-8")


def _lire_valeur(source, type_valeur: int) -> Any:
    if type_valeur == TYPES_VALEUR["uint32"]:
        return struct.unpack("<I", source.read(4))[0]
    if type_valeur == TYPES_VALEUR["int32"]:
        return struct.unpack("<i", source.read(4))[0]
    if type_valeur == TYPES_VALEUR["int64"]:
        return struct.unpack("<q", source.read(8))[0]
    if type_valeur == TYPES_VALEUR["uint64"]:
        return struct.unpack("<Q", source.read(8))[0]
    if type_valeur == TYPES_VALEUR["float32"]:
        return struct.unpack("<f", source.read(4))[0]
    if type_valeur == TYPES_VALEUR["bool"]:
        return struct.unpack("<?", source.read(1))[0]
    if type_valeur == TYPES_VALEUR["string"]:
        return _lire_chaine(source)
    if type_valeur == TYPES_VALEUR["array"]:
        (type_element,) = struct.unpack("<I", source.read(4))
        (nombre,) = struct.unpack("<Q", source.read(8))
        return [_lire_valeur(source, type_element) for _ in range(nombre)]
    raise ValueError(f"Type de valeur GGUF inconnu : {type_valeur}")


def lire_gguf(chemin: str | Path) -> dict:
    """Relit l'en-tête : version, métadonnées, description des tenseurs.

    Ne charge aucun poids — seulement de quoi vérifier qu'un fichier est bien
    formé et qu'il annonce ce qu'on croit.
    """
    with open(chemin, "rb") as source:
        if source.read(4) != MAGIE:
            raise ValueError(f"{chemin} : ce n'est pas un fichier GGUF.")
        (version,) = struct.unpack("<I", source.read(4))
        (n_tenseurs,) = struct.unpack("<Q", source.read(8))
        (n_metadonnees,) = struct.unpack("<Q", source.read(8))

        metadonnees = {}
        for _ in range(n_metadonnees):
            cle = _lire_chaine(source)
            (type_valeur,) = struct.unpack("<I", source.read(4))
            metadonnees[cle] = _lire_valeur(source, type_valeur)

        tenseurs = {}
        for _ in range(n_tenseurs):
            nom = _lire_chaine(source)
            (n_dimensions,) = struct.unpack("<I", source.read(4))
            formes = [struct.unpack("<Q", source.read(8))[0] for _ in range(n_dimensions)]
            (code,) = struct.unpack("<I", source.read(4))
            (decalage,) = struct.unpack("<Q", source.read(8))
            tenseurs[nom] = {"forme": formes, "type": code, "decalage": decalage}

    return {"version": version, "metadonnees": metadonnees, "tenseurs": tenseurs}


# --------------------------------------------------------------------------- #
# Correspondance avec l'architecture llama
# --------------------------------------------------------------------------- #

# Notre nom -> nom llama.cpp. Un pour un : c'est la même architecture.
CORRESPONDANCE = {
    "embeddings.weight": "token_embd.weight",
    "tete.weight": "output.weight",
    "norme_finale.weight": "output_norm.weight",
    "norme_attn.weight": "attn_norm.weight",
    "attn.q_proj.weight": "attn_q.weight",
    "attn.k_proj.weight": "attn_k.weight",
    "attn.v_proj.weight": "attn_v.weight",
    "attn.o_proj.weight": "attn_output.weight",
    "norme_mlp.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


def tenseurs_depuis_etat(etat: dict, demi_precision: bool = True) -> dict:
    """Traduit un état de modèle Futo en tenseurs nommés à la mode llama.cpp."""
    import numpy as np

    # Poids liés : `tete.weight` et `embeddings.weight` désignent le même
    # stockage. Réécrire les deux doublerait la taille du fichier pour rien —
    # llama.cpp retombe sur `token_embd.weight` quand `output.weight` manque.
    lies = (
        "tete.weight" in etat
        and "embeddings.weight" in etat
        and etat["tete.weight"].data_ptr() == etat["embeddings.weight"].data_ptr()
    )

    sortie = {}
    inconnus = []
    for nom, valeur in etat.items():
        if lies and nom == "tete.weight":
            continue
        tableau = valeur.detach().to("cpu").float().numpy()
        if demi_precision:
            tableau = tableau.astype(np.float16)

        if nom in CORRESPONDANCE:
            sortie[CORRESPONDANCE[nom]] = tableau
            continue
        if nom.startswith("blocs."):
            _, numero, reste = nom.split(".", 2)
            if reste in CORRESPONDANCE:
                sortie[f"blk.{numero}.{CORRESPONDANCE[reste]}"] = tableau
                continue
        inconnus.append(nom)

    if inconnus:
        raise KeyError(
            "Tenseurs sans correspondance llama.cpp : "
            + ", ".join(sorted(inconnus))
            + ". Le modèle a changé de forme, la table CORRESPONDANCE doit suivre."
        )
    return sortie


def metadonnees_depuis_config(cfg, tokenizer=None, nom: str = "futo") -> dict:
    """Métadonnées d'architecture, plus le vocabulaire si un tokenizer est fourni."""
    m = cfg.model
    metadonnees = {
        "general.architecture": "llama",
        "general.name": nom,
        "general.file_type": 1,  # 1 = tous les poids en F16
        "llama.block_count": m.n_layer,
        "llama.context_length": m.block_size,
        "llama.embedding_length": m.d_model,
        "llama.feed_forward_length": m.d_ff,
        "llama.attention.head_count": m.n_head,
        "llama.attention.head_count_kv": m.n_kv_head,
        "llama.attention.layer_norm_rms_epsilon": float(m.norm_eps),
        "llama.rope.dimension_count": m.head_dim,
        "llama.rope.freq_base": float(m.rope_theta),
        "llama.vocab_size": m.vocab_size,
    }
    if tokenizer is not None:
        metadonnees.update(_metadonnees_tokenizer(tokenizer))
    return metadonnees


def _metadonnees_tokenizer(tokenizer) -> dict:
    """Vocabulaire et fusions, lus directement dans le fichier du tokenizer.

    Le type annoncé est `gpt2`, c'est-à-dire BPE au niveau octet — ce que nous
    entraînons réellement. Voir la réserve en tête de module : llama.cpp
    n'appliquera PAS notre découpe préalable française.
    """
    donnees = json.loads(Path(tokenizer.chemin).read_text(encoding="utf-8"))
    modele = donnees.get("model", {})
    vocabulaire = modele.get("vocab", {})
    jetons = [None] * len(vocabulaire)
    for jeton, identifiant in vocabulaire.items():
        jetons[identifiant] = jeton

    fusions = modele.get("merges", [])
    if fusions and isinstance(fusions[0], list):
        fusions = [" ".join(paire) for paire in fusions]

    # 1 = jeton normal, 3 = jeton de contrôle, dans la nomenclature GGUF.
    ajoutes = {j["id"] for j in donnees.get("added_tokens", [])}
    types = [3 if i in ajoutes else 1 for i in range(len(jetons))]

    return {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "default",
        "tokenizer.ggml.tokens": jetons,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": list(fusions),
        "tokenizer.ggml.bos_token_id": tokenizer.id_debut,
        "tokenizer.ggml.eos_token_id": tokenizer.id_fin,
        "tokenizer.ggml.padding_token_id": tokenizer.id_rembourrage,
    }


def exporter_gguf(
    checkpoint: str | Path,
    sortie: str | Path,
    tokenizer=None,
    demi_precision: bool = True,
    nom: str | None = None,
) -> Path:
    """Convertit un point de reprise Futo en fichier GGUF."""
    import torch

    charge = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = charge["config_objet"]
    etat = charge["modele"] if "modele" in charge else charge["model"]
    etat = {c.replace("_orig_mod.", "").replace("module.", ""): v for c, v in etat.items()}

    tenseurs = tenseurs_depuis_etat(etat, demi_precision=demi_precision)
    metadonnees = metadonnees_depuis_config(cfg, tokenizer, nom or cfg.nom)

    # llama.cpp compare le nombre de jetons embarqués à la dimension des
    # embeddings et refuse le fichier s'ils divergent. Mieux vaut échouer ici,
    # avec un message qui dit quoi faire, qu'au chargement sur un téléphone.
    jetons = metadonnees.get("tokenizer.ggml.tokens")
    if jetons is not None:
        attendu = tenseurs["token_embd.weight"].shape[0]
        if len(jetons) != attendu:
            raise ValueError(
                f"Le tokenizer porte {len(jetons)} jetons, le modèle en attend {attendu}. "
                f"Ce GGUF serait refusé au chargement. Exportez avec le tokenizer "
                f"qui a servi à l'entraînement, ou passez --sans-tokenizer."
            )

    if not demi_precision:
        metadonnees["general.file_type"] = 0
    return ecrire_gguf(sortie, metadonnees, tenseurs)
