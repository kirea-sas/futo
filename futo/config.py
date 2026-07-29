"""Configuration typée de Futo.

Tout se règle par des `dataclass` typées, chargeables depuis un fichier YAML et
surchargeables en ligne de commande (`--set model.n_layer=12`). La configuration
complète est enregistrée dans chaque checkpoint : un checkpoint est donc
autoportant, on peut le recharger sans le fichier YAML d'origine.

Trois règles :

1. Une seule source de vérité — les `dataclass` ci-dessous. Le YAML ne fait que
   les remplir ; toute clé inconnue est une erreur (une faute de frappe dans un
   nom d'hyperparamètre ne doit jamais passer silencieusement).
2. Les types sont convertis d'après l'annotation du champ, pas devinés. `--set
   train.lr=3e-4` donne bien un `float`, `--set model.tie_weights=false` un `bool`.
3. Tout est sérialisable en dictionnaire de types simples, pour le YAML comme
   pour le checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "EvalConfig",
    "FutoConfig",
    "charger_config",
    "appliquer_surcharges",
]


# --------------------------------------------------------------------------- #
# Les blocs de configuration
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    """Architecture du transformeur décodeur."""

    # Vocabulaire et contexte
    vocab_size: int = 32_768
    block_size: int = 1_024  # longueur de contexte, en tokens

    # Profondeur et largeur
    n_layer: int = 12
    n_head: int = 12  # têtes de requête
    n_kv_head: int = 4  # têtes clé/valeur (GQA) ; doit diviser n_head
    d_model: int = 768
    d_ff: int | None = None  # None = calculé (SwiGLU, ~8/3 d_model arrondi)
    d_ff_multiple: int = 64  # arrondi de d_ff à un multiple de cette valeur

    # Détails du bloc
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_weights: bool = True  # embedding d'entrée == matrice de sortie
    dropout: float = 0.0  # 0 pendant le pré-entraînement (on ne fait qu'une passe)

    # Initialisation
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) doit être un multiple de n_kv_head "
                f"({self.n_kv_head}) : chaque groupe de têtes de requête partage "
                f"une tête clé/valeur."
            )
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) doit être divisible par n_head ({self.n_head})."
            )
        if self.d_ff is None:
            # SwiGLU utilise trois matrices au lieu de deux : à budget de
            # paramètres égal, la dimension cachée vaut 2/3 de celle d'un MLP
            # classique en 4·d_model, soit 8/3·d_model. On arrondit vers le haut
            # à un multiple de d_ff_multiple pour rester efficace sur GPU.
            brut = int(8 * self.d_model / 3)
            m = self.d_ff_multiple
            self.d_ff = ((brut + m - 1) // m) * m
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout doit être dans [0, 1[, reçu {self.dropout}.")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def nombre_parametres(self, avec_embeddings: bool = True) -> int:
        """Nombre exact de paramètres, calculé analytiquement.

        Sert à vérifier une configuration *sans* instancier le modèle (utile
        pour dimensionner un entraînement avant de louer un GPU). Le test
        `test_modele.py::test_comptage_parametres_analytique` vérifie que ce
        calcul colle au nombre réel, au paramètre près.

        Détail par couche :
          - attention : Wq (d·h·dh) + Wk (d·hkv·dh) + Wv (d·hkv·dh) + Wo (h·dh·d)
          - MLP SwiGLU : W_gate (d·f) + W_up (d·f) + W_down (f·d)
          - deux RMSNorm : 2·d
        Plus, hors couches : l'embedding (V·d), la norme finale (d) et la tête
        de sortie (V·d, mutualisée avec l'embedding si tie_weights).
        """
        d, f = self.d_model, self.d_ff
        assert f is not None  # garanti par __post_init__
        dh = self.head_dim

        attn = d * self.n_head * dh + 2 * (d * self.n_kv_head * dh) + self.n_head * dh * d
        mlp = 3 * d * f
        normes = 2 * d
        par_couche = attn + mlp + normes

        total = self.n_layer * par_couche + d  # + norme finale
        if avec_embeddings:
            total += self.vocab_size * d  # embedding d'entrée
            if not self.tie_weights:
                total += self.vocab_size * d  # tête de sortie distincte
        return total

    def flops_par_token(self) -> float:
        """FLOPs d'une passe avant+arrière par token, pour le calcul du MFU.

        Convention de l'annexe B de Kaplan et al. (2020), reprise par PaLM :
        une passe avant coûte ≈ 2·N multiplications-additions par token (N =
        nombre de paramètres hors embeddings), l'arrière en coûte le double,
        d'où le facteur 6·N. À quoi s'ajoute l'attention elle-même, quadratique
        en la longueur de contexte et absente du comptage de paramètres :
        12·n_layer·d_model·block_size par token (6 pour QKᵀ, 6 pour A·V, arrière
        compris).

        On exclut les embeddings de N : la table de lookup ne fait pas de
        multiplication matricielle. La tête de sortie, elle, en fait une — on
        la garde donc.
        """
        d = self.d_model
        assert self.d_ff is not None
        dh = self.head_dim
        attn = d * self.n_head * dh + 2 * (d * self.n_kv_head * dh) + self.n_head * dh * d
        mlp = 3 * d * self.d_ff
        n_denses = self.n_layer * (attn + mlp) + self.vocab_size * d  # + tête de sortie
        cout_attention = 12 * self.n_layer * d * self.block_size
        return 6.0 * n_denses + cout_attention


@dataclass
class DataConfig:
    """Où sont les données et comment on les lit."""

    # Dossier contenant les shards binaires produits par `futo data preparer`
    dossier: str = "data/prepare"
    # Préfixes des shards d'entraînement et de validation
    prefixe_train: str = "train"
    prefixe_val: str = "val"
    # Chemin du tokenizer (fichier tokenizer.json)
    tokenizer: str = "data/tokenizer/futo-tokenizer.json"
    # Nombre de séquences tirées par lot (le lot en tokens vaut batch_size ×
    # block_size × grad_accum × world_size)
    batch_size: int = 8
    # Graine du tirage des positions dans les shards
    seed: int = 1234


@dataclass
class TrainConfig:
    """Boucle d'entraînement."""

    # Durée
    max_steps: int = 10_000
    grad_accum: int = 1

    # Optimiseur (AdamW)
    lr: float = 6e-4
    lr_min_ratio: float = 0.1  # lr_min = lr × ce ratio
    warmup_steps: int = 200
    schedule: str = "cosine"  # "cosine" | "wsd" | "constant"
    decay_steps: int | None = None  # WSD : durée de la descente finale
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Précision et performance
    dtype: str = "bf16"  # "bf16" | "fp16" | "fp32"
    compile: bool = False
    gradient_checkpointing: bool = False

    # Journal et sauvegardes
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 50
    save_every: int = 1_000
    dossier_sortie: str = "sorties/run"
    garder_n_checkpoints: int = 3

    # Reprise
    reprendre: str | None = None  # chemin d'un checkpoint, ou "auto"

    # Reproductibilité
    seed: int = 1337
    deterministe: bool = False  # désactive les noyaux non déterministes (plus lent)

    def __post_init__(self) -> None:
        if self.schedule not in {"cosine", "wsd", "constant"}:
            raise ValueError(
                f"schedule inconnu : {self.schedule!r} "
                f"(attendu : cosine, wsd ou constant)."
            )
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"dtype inconnu : {self.dtype!r} (attendu : bf16, fp16 ou fp32).")
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) doit être < max_steps ({self.max_steps})."
            )


@dataclass
class EvalConfig:
    """Évaluation."""

    batches: int = 100
    # Sondes grammaticales françaises (paires minimales) — voir futo/eval.py
    sondes: str = "data/sondes/paires-minimales-fr.jsonl"


@dataclass
class FutoConfig:
    """Configuration complète d'une expérience."""

    nom: str = "futo-tiny"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FutoConfig:
        return _construire(cls, d)

    def tokens_par_pas(self, world_size: int = 1) -> int:
        """Nombre de tokens vus à chaque pas d'optimisation."""
        return (
            self.data.batch_size
            * self.model.block_size
            * self.train.grad_accum
            * world_size
        )

    def resume(self, world_size: int = 1) -> str:
        """Résumé lisible, affiché au démarrage d'un entraînement."""
        m = self.model
        n = m.nombre_parametres()
        n_sans_emb = m.nombre_parametres(avec_embeddings=False)
        tpp = self.tokens_par_pas(world_size)
        lignes = [
            f"Configuration « {self.nom} »",
            f"  {m.n_layer} couches · d_model {m.d_model} · {m.n_head} têtes "
            f"({m.n_kv_head} kv, dim {m.head_dim}) · d_ff {m.d_ff}",
            f"  vocabulaire {m.vocab_size} · contexte {m.block_size} tokens"
            f" · poids liés : {'oui' if m.tie_weights else 'non'}",
            f"  paramètres : {n / 1e6:.1f} M (dont {n_sans_emb / 1e6:.1f} M hors embeddings)",
            f"  lot : {tpp:,} tokens/pas".replace(",", " "),
            f"  budget : {self.train.max_steps} pas ≈ "
            f"{self.train.max_steps * tpp / 1e9:.2f} G tokens",
        ]
        return "\n".join(lignes)


# --------------------------------------------------------------------------- #
# Chargement, fusion et surcharge
# --------------------------------------------------------------------------- #


def _est_optionnel(annotation: Any) -> tuple[bool, Any]:
    """Déplie `X | None` et renvoie (est_optionnel, X).

    Deux écritures coexistent : `Optional[X]`, dont l'origine est
    `typing.Union`, et `X | None` (Python 3.10+), dont l'origine est
    `types.UnionType`. Il faut reconnaître les deux.
    """
    origine = typing.get_origin(annotation)
    if origine is typing.Union or origine is types.UnionType:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return True, args[0]
    return False, annotation


def _convertir(valeur: Any, annotation: Any, chemin: str) -> Any:
    """Convertit `valeur` vers le type annoté. Lève ValueError si impossible."""
    optionnel, brut = _est_optionnel(annotation)
    if valeur is None:
        if optionnel:
            return None
        raise ValueError(f"{chemin} : None interdit pour le type {annotation}.")

    if brut is bool:
        if isinstance(valeur, bool):
            return valeur
        texte = str(valeur).strip().lower()
        if texte in {"1", "true", "vrai", "oui", "yes", "on"}:
            return True
        if texte in {"0", "false", "faux", "non", "no", "off"}:
            return False
        raise ValueError(f"{chemin} : booléen attendu, reçu {valeur!r}.")
    if brut is int:
        if isinstance(valeur, bool):
            raise ValueError(f"{chemin} : entier attendu, reçu un booléen.")
        # Autorise "1e4" et 1.0 tant que la valeur est entière.
        f = float(valeur)
        if f != int(f):
            raise ValueError(f"{chemin} : entier attendu, reçu {valeur!r}.")
        return int(f)
    if brut is float:
        return float(valeur)
    if brut is str:
        return str(valeur)
    return valeur


def _construire(cls: type, d: dict[str, Any], chemin: str = "") -> Any:
    """Instancie récursivement une dataclass depuis un dictionnaire."""
    if not isinstance(d, dict):
        raise ValueError(f"{chemin or '<racine>'} : dictionnaire attendu, reçu {type(d).__name__}.")

    connus = {f.name: f for f in fields(cls)}
    inconnus = sorted(set(d) - set(connus))
    if inconnus:
        proches = ", ".join(sorted(connus))
        raise ValueError(
            f"{chemin or '<racine>'} : clé(s) inconnue(s) {inconnus}. "
            f"Clés acceptées : {proches}."
        )

    kwargs: dict[str, Any] = {}
    for nom, f in connus.items():
        if nom not in d:
            continue
        sous_chemin = f"{chemin}.{nom}" if chemin else nom
        annotation = f.type
        if isinstance(annotation, str):
            # `from __future__ import annotations` transforme les annotations en
            # chaînes : on les résout dans le module de la dataclass.
            annotation = typing.get_type_hints(cls)[nom]
        if is_dataclass(annotation) and isinstance(annotation, type):
            kwargs[nom] = _construire(annotation, d[nom], sous_chemin)
        else:
            kwargs[nom] = _convertir(d[nom], annotation, sous_chemin)
    return cls(**kwargs)


def _fusionner(base: dict[str, Any], surcouche: dict[str, Any]) -> dict[str, Any]:
    """Fusion récursive : `surcouche` écrase `base`, dictionnaire par dictionnaire."""
    resultat = dict(base)
    for cle, valeur in surcouche.items():
        if cle in resultat and isinstance(resultat[cle], dict) and isinstance(valeur, dict):
            resultat[cle] = _fusionner(resultat[cle], valeur)
        else:
            resultat[cle] = valeur
    return resultat


def appliquer_surcharges(brut: dict[str, Any], surcharges: list[str]) -> dict[str, Any]:
    """Applique des surcharges `chemin.pointe=valeur` venues de la ligne de commande.

    La valeur est d'abord lue comme du JSON (ce qui gère 12, 3e-4, true, null,
    [1,2]) ; si ce n'est pas du JSON valide, elle est gardée telle quelle en
    chaîne, et c'est `_convertir` qui tranchera d'après le type du champ.
    """
    resultat = dict(brut)
    for surcharge in surcharges:
        if "=" not in surcharge:
            raise ValueError(
                f"Surcharge mal formée : {surcharge!r}. Forme attendue : chemin.cle=valeur"
            )
        chemin, _, texte = surcharge.partition("=")
        chemin = chemin.strip()
        if not chemin:
            raise ValueError(f"Surcharge mal formée : {surcharge!r} (chemin vide).")
        try:
            valeur: Any = json.loads(texte)
        except json.JSONDecodeError:
            valeur = texte

        morceaux = chemin.split(".")
        noeud = resultat
        for morceau in morceaux[:-1]:
            suivant = noeud.get(morceau)
            if not isinstance(suivant, dict):
                suivant = {}
                noeud[morceau] = suivant
            noeud = suivant
        noeud[morceaux[-1]] = valeur
    return resultat


def charger_config(
    chemin: str | Path | None = None,
    surcharges: list[str] | None = None,
) -> FutoConfig:
    """Charge une configuration YAML, applique les surcharges, valide le tout.

    Une configuration peut hériter d'une autre via la clé `hérite` (ou
    `herite`), ce qui évite de répéter les hyperparamètres communs entre
    `futo-small.yaml` et `futo-base.yaml`. Le chemin hérité est résolu
    relativement au fichier qui l'invoque.
    """
    brut: dict[str, Any] = {}
    if chemin is not None:
        brut = _lire_yaml_avec_heritage(Path(chemin), vus=[])
    if surcharges:
        brut = appliquer_surcharges(brut, surcharges)
    return FutoConfig.from_dict(brut)


def _lire_yaml_avec_heritage(chemin: Path, vus: list[Path]) -> dict[str, Any]:
    import yaml

    chemin = chemin.resolve()
    if chemin in vus:
        chaine = " → ".join(p.name for p in [*vus, chemin])
        raise ValueError(f"Héritage circulaire entre configurations : {chaine}")
    if not chemin.exists():
        raise FileNotFoundError(f"Configuration introuvable : {chemin}")

    with chemin.open("r", encoding="utf-8") as fh:
        contenu = yaml.safe_load(fh) or {}
    if not isinstance(contenu, dict):
        raise ValueError(f"{chemin} : la racine du YAML doit être un dictionnaire.")

    parent_cle = "hérite" if "hérite" in contenu else ("herite" if "herite" in contenu else None)
    if parent_cle is None:
        return contenu

    parent_chemin = chemin.parent / str(contenu.pop(parent_cle))
    parent = _lire_yaml_avec_heritage(parent_chemin, [*vus, chemin])
    return _fusionner(parent, contenu)
