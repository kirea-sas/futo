"""La boucle d'entraînement.

Trois exigences ont guidé l'écriture, dans cet ordre :

1. **Ne jamais perdre un entraînement.** Un pré-entraînement dure des jours et
   coûte des centaines d'euros de location GPU ; un plantage à 80 % ne doit
   jamais obliger à tout reprendre. Les checkpoints enregistrent de quoi
   reprendre à l'identique — poids, optimiseur, pas, et l'état des trois
   générateurs aléatoires (PyTorch, NumPy, Python).
2. **Voir ce qui se passe.** Perte, débit en tokens/s, MFU, norme du gradient et
   pas de temps sont journalisés en JSONL *et* en CSV, sur le disque local :
   aucun service en ligne n'est requis, et le journal reste lisible avec un
   tableur si besoin.
3. **Rester lisible.** Une seule fonction `entrainer`, sans couche
   d'abstraction : on doit pouvoir suivre le flux du début à la fin.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import FutoConfig
from .data import ChargeurTokens
from .model import Futo

__all__ = [
    "entrainer",
    "taux_apprentissage",
    "sauvegarder",
    "charger",
    "Journal",
    "choisir_peripherique",
    "nom_du_materiel",
    "flops_crete_du_materiel",
    "contexte_autocast",
    "mesurer_debit",
]


# --------------------------------------------------------------------------- #
# Planning du taux d'apprentissage
# --------------------------------------------------------------------------- #


def taux_apprentissage(pas: int, cfg: FutoConfig) -> float:
    """Le taux d'apprentissage au pas donné.

    Trois formes, toutes précédées du même échauffement linéaire :

    * ``cosine`` — la référence. Descend en cosinus jusqu'à `lr × lr_min_ratio`.
      Il faut connaître la durée totale à l'avance.
    * ``wsd`` — échauffement, plateau, puis descente finale (*warmup-stable-decay*).
      Son intérêt pratique : le plateau peut être prolongé sans rien recalculer,
      donc on peut décider d'allonger un entraînement en cours de route, ce que le
      cosinus interdit.
    * ``constant`` — pour déboguer, ou pour un test qui doit rester prévisible.

    L'échauffement n'est pas une coquetterie : au premier pas, les statistiques
    d'Adam sont vides et le gradient est énorme. Démarrer au taux plein fait
    diverger presque à coup sûr.
    """
    t = cfg.train
    if pas < t.warmup_steps:
        # +1 pour que le tout premier pas ne soit pas à un taux nul.
        return t.lr * (pas + 1) / max(t.warmup_steps, 1)

    lr_min = t.lr * t.lr_min_ratio

    if t.schedule == "constant":
        return t.lr

    if t.schedule == "cosine":
        avance = (pas - t.warmup_steps) / max(t.max_steps - t.warmup_steps, 1)
        avance = min(max(avance, 0.0), 1.0)
        return lr_min + 0.5 * (t.lr - lr_min) * (1.0 + math.cos(math.pi * avance))

    # WSD : plateau puis descente linéaire sur les `decay_steps` derniers pas.
    decay = t.decay_steps if t.decay_steps is not None else max(t.max_steps // 10, 1)
    debut_descente = t.max_steps - decay
    if pas < debut_descente:
        return t.lr
    avance = (pas - debut_descente) / max(decay, 1)
    return t.lr + (lr_min - t.lr) * min(avance, 1.0)


# --------------------------------------------------------------------------- #
# Matériel
# --------------------------------------------------------------------------- #

# FLOPs crête en bf16/fp16 (calcul dense, sans parcimonie), d'après les fiches
# constructeur. Servent uniquement au calcul du MFU ; un chiffre absent donne un
# MFU nul, jamais une erreur.
#
# Les valeurs Apple sont des ORDRES DE GRANDEUR. Apple ne publie pas de chiffre
# de FLOPs crête comparable à celui des cartes NVIDIA ; ceux-ci sont déduits du
# nombre de cœurs GPU et de la fréquence. Le MFU affiché sur un Mac est donc
# indicatif, et ne doit pas être comparé directement à celui d'un H100.
FLOPS_CRETE = {
    "H100": 989e12,
    "H200": 989e12,
    "A100": 312e12,
    "L40S": 362e12,
    "A6000": 155e12,
    "4090": 165e12,
    "4080": 98e12,
    "3090": 71e12,
    "V100": 125e12,
    # Apple Silicon — approximatif, voir ci-dessus.
    "M4 Max": 18e12,
    "M4 Pro": 9e12,
    "M3 Ultra": 28e12,
    "M3 Max": 14e12,
    "M3 Pro": 7e12,
    "M2 Ultra": 27e12,
    "M2 Max": 13.6e12,
    "M2 Pro": 7e12,
    "M1 Ultra": 21e12,
    "M1 Max": 10.4e12,
    "M1 Pro": 5.2e12,
    "M4": 5e12,
    "M3": 4e12,
    "M2": 3.6e12,
    "M1": 2.6e12,
}


def nom_du_materiel(peripherique: torch.device | None = None) -> str:
    """Nom lisible de l'accélérateur : « NVIDIA H100 », « Apple M3 Max », « processeur »."""
    if peripherique is None:
        peripherique = choisir_peripherique()
    if peripherique.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if peripherique.type == "mps":
        # PyTorch n'expose pas le nom de la puce : on le demande au système.
        # Lecture seule, et sans conséquence si la commande n'existe pas.
        import subprocess

        try:
            sortie = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2, check=True,
            )
            if sortie.stdout.strip():
                return sortie.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return "Apple Silicon (puce non identifiée)"
    return "processeur"


def flops_crete_du_materiel(nom: str | None = None) -> float:
    """Devine les FLOPs crête à partir du nom du matériel.

    Renvoie 0 si le matériel est inconnu : le MFU s'affiche alors comme nul,
    plutôt que faux. Les clés sont essayées de la plus longue à la plus courte,
    sinon « M3 » masquerait « M3 Max ».
    """
    if nom is None:
        nom = nom_du_materiel()
    nom_bas = nom.lower()
    for cle in sorted(FLOPS_CRETE, key=len, reverse=True):
        if cle.lower() in nom_bas:
            return FLOPS_CRETE[cle]
    return 0.0


def choisir_peripherique() -> torch.device:
    """CUDA si présent, sinon le GPU intégré d'un Mac (MPS), sinon le processeur."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def contexte_autocast(peripherique: torch.device, dtype: torch.dtype, dire=print):
    """Construit le contexte de précision mixte, avec un repli *vérifié*.

    Le cas délicat est celui du Mac : la prise en charge de l'autocast sur MPS
    dépend de la version de PyTorch et du type demandé, et une combinaison non
    gérée échoue au premier pas — donc après le chargement du modèle et des
    données. Plutôt que de le supposer, on l'essaie ici sur un tenseur
    minuscule ; si ça ne passe pas, on retombe en float32 en le disant.
    """
    if dtype == torch.float32:
        return nullcontext()
    try:
        with torch.autocast(device_type=peripherique.type, dtype=dtype):
            a = torch.ones(2, 2, device=peripherique)
            _ = a @ a
    except (RuntimeError, ValueError, NotImplementedError, TypeError) as e:
        dire(
            f"  autocast {str(dtype).replace('torch.', '')} indisponible sur "
            f"{peripherique.type} — repli sur float32 ({str(e)[:60]})."
        )
        return nullcontext()
    return torch.autocast(device_type=peripherique.type, dtype=dtype)


def resoudre_dtype(demande: str, peripherique: torch.device, dire=print) -> torch.dtype:
    """Retient le meilleur type disponible, en annonçant tout repli.

    Un repli silencieux se paierait en heures de calcul inexpliquées.
    """
    if demande == "fp32":
        return torch.float32
    if demande == "bf16":
        if peripherique.type == "cuda" and not torch.cuda.is_bf16_supported():
            dire("  bf16 non pris en charge par ce GPU, repli sur fp16.")
            return torch.float16
        return torch.bfloat16
    return torch.float16


# --------------------------------------------------------------------------- #
# Journal
# --------------------------------------------------------------------------- #


class Journal:
    """Écrit les mesures en JSONL et en CSV, sans dépendance ni réseau.

    Le JSONL sert aux outils, le CSV au tableur. Les deux fichiers sont vidés
    (`flush`) à chaque ligne : un plantage ne fait donc jamais perdre le journal.
    """

    COLONNES = [
        "pas",
        "perte",
        "perte_val",
        "lr",
        "norme_grad",
        "tokens_vus",
        "tokens_par_s",
        "mfu",
        "secondes",
    ]

    def __init__(self, dossier: str | Path, actif: bool = True) -> None:
        self.actif = actif
        if not actif:
            return
        dossier = Path(dossier)
        dossier.mkdir(parents=True, exist_ok=True)
        self._jsonl = (dossier / "journal.jsonl").open("a", encoding="utf-8")
        chemin_csv = dossier / "journal.csv"
        nouveau = not chemin_csv.exists() or chemin_csv.stat().st_size == 0
        self._csv = chemin_csv.open("a", encoding="utf-8")
        if nouveau:
            self._csv.write(",".join(self.COLONNES) + "\n")
            self._csv.flush()

    def ecrire(self, **mesures) -> None:
        if not self.actif:
            return
        self._jsonl.write(json.dumps(mesures, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        ligne = []
        for colonne in self.COLONNES:
            valeur = mesures.get(colonne, "")
            ligne.append(f"{valeur:.6g}" if isinstance(valeur, float) else str(valeur))
        self._csv.write(",".join(ligne) + "\n")
        self._csv.flush()

    def fermer(self) -> None:
        if not self.actif:
            return
        self._jsonl.close()
        self._csv.close()


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def _etat_alea() -> dict:
    """Photographie des trois générateurs aléatoires du processus.

    Les états sont explicitement ramenés sur le processeur : c'est là qu'ils
    vivent, et c'est là qu'il faudra les rendre au retour (voir
    `_octets_sur_processeur`).
    """
    etat = {
        "torch": torch.get_rng_state().cpu(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        etat["cuda"] = [e.cpu() for e in torch.cuda.get_rng_state_all()]
    return etat


def _octets_sur_processeur(tenseur: torch.Tensor) -> torch.Tensor:
    """Ramène un état de générateur sur le processeur, en octets.

    C'est le correctif d'un défaut que seule une vraie carte pouvait révéler.
    Un checkpoint se recharge avec `map_location=<périphérique>`, et
    `map_location` déplace **tous** les tenseurs du fichier — y compris l'état
    du générateur aléatoire, qui n'a pourtant rien à faire sur un accélérateur.
    `torch.set_rng_state` exige un ByteTensor du processeur et refuse le reste :

        TypeError: RNG state must be a torch.ByteTensor

    La reprise était donc cassée sur tout accélérateur — MPS comme CUDA — et
    fonctionnait uniquement sur processeur, c'est-à-dire précisément là où la
    suite de tests s'exécute. Signalé par un run sur Apple M2 Max.
    """
    if tenseur.device.type != "cpu":
        tenseur = tenseur.cpu()
    if tenseur.dtype != torch.uint8:
        tenseur = tenseur.to(torch.uint8)
    return tenseur


def _restaurer_alea(etat: dict) -> None:
    torch.set_rng_state(_octets_sur_processeur(etat["torch"]))
    np.random.set_state(etat["numpy"])
    random.setstate(etat["python"])
    if "cuda" in etat and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_octets_sur_processeur(e) for e in etat["cuda"]])


def sauvegarder(
    chemin: str | Path,
    modele: Futo,
    optimiseur: torch.optim.Optimizer,
    cfg: FutoConfig,
    pas: int,
    meilleure_val: float,
    scaler=None,
) -> None:
    """Enregistre tout ce qu'il faut pour reprendre à l'identique.

    Écriture atomique : on écrit dans un fichier temporaire puis on le renomme.
    Sans cela, un plantage *pendant* la sauvegarde laisserait un checkpoint
    tronqué — et c'est justement le moment où l'on plante le plus, le disque
    étant sollicité.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    brut = modele
    # Déballe les enveloppes DDP / torch.compile pour que le checkpoint reste
    # chargeable dans un processus mono-GPU sans compilation.
    brut = getattr(brut, "module", brut)
    brut = getattr(brut, "_orig_mod", brut)

    charge = {
        "version": 1,
        "pas": pas,
        "config": cfg.to_dict(),
        "modele": brut.state_dict(),
        "optimiseur": optimiseur.state_dict(),
        "meilleure_val": meilleure_val,
        "alea": _etat_alea(),
    }
    if scaler is not None:
        charge["scaler"] = scaler.state_dict()

    temporaire = chemin.with_suffix(chemin.suffix + ".tmp")
    torch.save(charge, temporaire)
    temporaire.replace(chemin)


def charger(
    chemin: str | Path,
    peripherique: torch.device | str = "cpu",
    restaurer_alea: bool = True,
) -> tuple[Futo, dict]:
    """Recharge un modèle depuis un checkpoint, avec sa configuration.

    Le checkpoint est autoportant : ni le fichier YAML ni le code appelant
    n'ont besoin de connaître les hyperparamètres.
    """
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {chemin}")
    charge = torch.load(chemin, map_location=peripherique, weights_only=False)

    cfg = FutoConfig.from_dict(charge["config"])
    modele = Futo(cfg.model)
    modele.load_state_dict(charge["modele"])
    modele.to(peripherique)

    if restaurer_alea and "alea" in charge:
        _restaurer_alea(charge["alea"])

    charge["config_objet"] = cfg
    return modele, charge


def _nettoyer_checkpoints(dossier: Path, garder: int) -> None:
    """Ne conserve que les `garder` checkpoints périodiques les plus récents.

    `meilleur.pt` et `dernier.pt` ne sont jamais supprimés.
    """
    if garder <= 0:
        return
    periodiques = sorted(
        dossier.glob("pas_*.pt"), key=lambda p: int(p.stem.split("_")[1])
    )
    for vieux in periodiques[:-garder]:
        vieux.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Évaluation intercalaire
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluer(modele, chargeur: ChargeurTokens, n_lots: int, peripherique, contexte) -> float:
    """Perte moyenne sur `n_lots` lots de validation.

    Les lots sont ceux des pas 0…n_lots-1 du chargeur de validation : ils sont
    donc *toujours les mêmes*, ce qui rend deux évaluations comparables. Une
    validation tirée au hasard à chaque fois ferait osciller la courbe sans
    que le modèle ait changé.
    """
    etait_en_entrainement = modele.training
    modele.eval()
    total = 0.0
    for i in range(n_lots):
        entree, cible = chargeur.lot(i)
        entree, cible = entree.to(peripherique), cible.to(peripherique)
        with contexte:
            _, perte = modele(entree, cibles=cible)
        total += perte.item()
    modele.train(etait_en_entrainement)
    return total / max(n_lots, 1)


# --------------------------------------------------------------------------- #
# Mesure de débit
# --------------------------------------------------------------------------- #


@dataclass
class ResultatDebit:
    """Ce que mesure `futo bench` : le seul chiffre qui permette de décider."""

    tokens_par_s: float
    mfu: float
    secondes_par_pas: float
    materiel: str
    dtype: str
    micro_lots: int
    tokens_mesures: int

    def duree_estimee(self, tokens_totaux: int) -> float:
        """Secondes pour un entraînement complet, au débit mesuré."""
        return tokens_totaux / max(self.tokens_par_s, 1e-9)


def _synchroniser(peripherique: torch.device) -> None:
    """Attend la fin des calculs en cours.

    Indispensable pour chronométrer : sur GPU comme sur MPS, les opérations sont
    lancées de façon asynchrone. Sans synchronisation, on mesurerait la vitesse
    à laquelle Python empile des ordres, pas celle à laquelle la machine calcule.
    """
    if peripherique.type == "cuda":
        torch.cuda.synchronize()
    elif peripherique.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def mesurer_debit(
    cfg: FutoConfig,
    micro_lots: int = 20,
    echauffement: int = 3,
    verbeux: bool = True,
) -> ResultatDebit:
    """Mesure le débit réel d'une configuration, sur des données synthétiques.

    Aucun corpus, aucun tokenizer, aucun fichier : des tokens tirés au hasard
    suffisent, puisque le coût de calcul d'un transformeur ne dépend pas du
    contenu. On exécute de vraies passes avant et arrière, avec le vrai
    optimiseur, dans la vraie précision — c'est bien le débit d'entraînement
    qu'on mesure, pas une approximation.

    À quoi ça sert : répondre à « combien de temps sur MA machine » en une
    trentaine de secondes, avant d'engager des jours de calcul ou des euros de
    location. Les estimations de `futo info` reposent sur des FLOPs crête
    théoriques ; celle-ci repose sur une mesure.
    """

    def dire(message: str = "") -> None:
        if verbeux:
            print(message, flush=True)

    peripherique = choisir_peripherique()
    dtype = resoudre_dtype(cfg.train.dtype, peripherique, dire)
    contexte = contexte_autocast(peripherique, dtype, dire)

    torch.manual_seed(cfg.train.seed)
    modele = Futo(cfg.model).to(peripherique)
    if cfg.train.gradient_checkpointing:
        modele.activer_gradient_checkpointing(True)
    modele.train()

    optimiseur = torch.optim.AdamW(
        modele.groupes_parametres(cfg.train.weight_decay),
        lr=cfg.train.lr,
        betas=(cfg.train.beta1, cfg.train.beta2),
        fused=peripherique.type == "cuda",
    )

    B, T = cfg.data.batch_size, cfg.model.block_size
    tokens_par_micro_lot = B * T
    # Un seul lot, réutilisé : on mesure le calcul, pas le chargement.
    entree = torch.randint(0, cfg.model.vocab_size, (B, T), device=peripherique)
    cible = torch.randint(0, cfg.model.vocab_size, (B, T), device=peripherique)

    def un_micro_lot(indice: int) -> None:
        with contexte:
            _, perte = modele(entree, cibles=cible)
            perte = perte / cfg.train.grad_accum
        perte.backward()
        if (indice + 1) % cfg.train.grad_accum == 0:
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(modele.parameters(), cfg.train.grad_clip)
            optimiseur.step()
            optimiseur.zero_grad(set_to_none=True)

    # L'échauffement ne compte pas : première allocation, compilation des
    # noyaux, montée en fréquence. Le mesurer fausserait tout vers le bas.
    dire(f"  échauffement ({echauffement} micro-lots)…")
    for i in range(echauffement):
        un_micro_lot(i)
    _synchroniser(peripherique)
    optimiseur.zero_grad(set_to_none=True)

    dire(f"  mesure ({micro_lots} micro-lots de {tokens_par_micro_lot} tokens)…")
    depart = time.perf_counter()
    for i in range(micro_lots):
        un_micro_lot(i)
    _synchroniser(peripherique)
    ecoule = time.perf_counter() - depart

    tokens_mesures = micro_lots * tokens_par_micro_lot
    tokens_par_s = tokens_mesures / max(ecoule, 1e-9)

    materiel = nom_du_materiel(peripherique)
    crete = flops_crete_du_materiel(materiel)
    mfu = cfg.model.flops_par_token() * tokens_par_s / crete if crete > 0 else 0.0

    return ResultatDebit(
        tokens_par_s=tokens_par_s,
        mfu=mfu,
        secondes_par_pas=cfg.tokens_par_pas() / max(tokens_par_s, 1e-9),
        materiel=materiel,
        dtype=str(dtype).replace("torch.", ""),
        micro_lots=micro_lots,
        tokens_mesures=tokens_mesures,
    )


# --------------------------------------------------------------------------- #
# La boucle
# --------------------------------------------------------------------------- #


@dataclass
class ResultatEntrainement:
    pas_effectues: int
    perte_finale: float
    meilleure_val: float
    tokens_vus: int
    secondes: float
    dossier: Path


def entrainer(cfg: FutoConfig, verbeux: bool = True) -> ResultatEntrainement:
    """Entraîne un modèle selon `cfg`, et renvoie un résumé du déroulement."""
    t = cfg.train

    # -- distribué ---------------------------------------------------------- #
    rang, monde, rang_local = 0, 1, 0
    distribue = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distribue:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rang = dist.get_rank()
        monde = dist.get_world_size()
        rang_local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(rang_local)
    maitre = rang == 0

    def dire(message: str = "") -> None:
        if verbeux and maitre:
            print(message, flush=True)

    # -- reproductibilité --------------------------------------------------- #
    graine = t.seed + rang  # un rang par graine : les dropouts diffèrent
    torch.manual_seed(graine)
    np.random.seed(graine % (2**32))
    random.seed(graine)
    if t.deterministe:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.cuda.is_available():
        # TF32 : division par ~3 du temps des matmuls sur Ampère et au-delà,
        # pour une perte de précision sans effet mesurable à l'entraînement.
        # Ces bascules n'existent que sur CUDA.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    peripherique = choisir_peripherique()
    if distribue:
        peripherique = torch.device(f"cuda:{rang_local}")
    dtype = resoudre_dtype(t.dtype, peripherique, dire)
    contexte = contexte_autocast(peripherique, dtype, dire)

    # -- données ------------------------------------------------------------ #
    chargeur = ChargeurTokens(
        cfg.data.dossier,
        cfg.data.prefixe_train,
        cfg.model.block_size,
        cfg.data.batch_size,
        seed=cfg.data.seed,
        rang=rang,
        monde=monde,
    )
    try:
        chargeur_val = ChargeurTokens(
            cfg.data.dossier,
            cfg.data.prefixe_val,
            cfg.model.block_size,
            cfg.data.batch_size,
            seed=cfg.data.seed + 1,
        )
    except FileNotFoundError:
        chargeur_val = None
        dire("  (aucun shard de validation : l'évaluation intercalaire est désactivée)")

    if chargeur.vocab_size != cfg.model.vocab_size:
        raise ValueError(
            f"Incohérence de vocabulaire : les shards ont été préparés avec "
            f"{chargeur.vocab_size} tokens, la configuration en annonce "
            f"{cfg.model.vocab_size}. Corrigez model.vocab_size, ou repréparez "
            f"les données avec le bon tokenizer."
        )

    # -- modèle ------------------------------------------------------------- #
    modele = Futo(cfg.model).to(peripherique)
    if t.gradient_checkpointing:
        modele.activer_gradient_checkpointing(True)

    optimiseur = torch.optim.AdamW(
        modele.groupes_parametres(t.weight_decay),
        lr=t.lr,
        betas=(t.beta1, t.beta2),
        eps=t.eps,
        # `fused` fait tout le pas d'optimisation dans un seul noyau CUDA :
        # nettement plus rapide dès que le modèle a beaucoup de tenseurs.
        fused=peripherique.type == "cuda",
    )
    # Le scaler ne sert qu'en fp16 sur CUDA : bf16 a le même exposant que fp32
    # et ne déborde pas, et la mise à l'échelle n'est pas prise en charge sur
    # MPS ni sur processeur. Désactivé, il se traverse sans rien faire.
    scaler = torch.amp.GradScaler(
        device=peripherique.type,
        enabled=(dtype == torch.float16 and peripherique.type == "cuda"),
    )

    pas_depart = 0
    meilleure_val = float("inf")
    dossier = Path(t.dossier_sortie)

    # -- reprise ------------------------------------------------------------ #
    chemin_reprise = t.reprendre
    if chemin_reprise == "auto":
        candidat = dossier / "dernier.pt"
        chemin_reprise = str(candidat) if candidat.exists() else None
        if chemin_reprise is None:
            dire("  (reprise « auto » : aucun checkpoint trouvé, on démarre de zéro)")
    if chemin_reprise:
        dire(f"  reprise depuis {chemin_reprise}")
        charge = torch.load(chemin_reprise, map_location=peripherique, weights_only=False)
        modele.load_state_dict(charge["modele"])
        optimiseur.load_state_dict(charge["optimiseur"])
        if "scaler" in charge:
            scaler.load_state_dict(charge["scaler"])
        pas_depart = charge["pas"]
        meilleure_val = charge.get("meilleure_val", float("inf"))
        if "alea" in charge:
            _restaurer_alea(charge["alea"])
        dire(f"  reprise au pas {pas_depart}")

    modele_calcul = modele
    if t.compile:
        dire("  compilation du modèle (le premier pas sera lent)…")
        modele_calcul = torch.compile(modele)
    if distribue:
        from torch.nn.parallel import DistributedDataParallel

        modele_calcul = DistributedDataParallel(modele_calcul, device_ids=[rang_local])

    # -- résumé ------------------------------------------------------------- #
    tokens_par_pas = cfg.tokens_par_pas(monde)
    if maitre:
        dire(cfg.resume(monde))
        dire(
            f"  matériel : {nom_du_materiel(peripherique)}"
            + (f" ×{monde}" if monde > 1 else "")
            + f" · {str(dtype).replace('torch.', '')}"
        )
        dire(f"  corpus : {len(chargeur):,} tokens".replace(",", " "))
        dire()

    journal = Journal(dossier, actif=maitre)
    depart_horloge = time.perf_counter()
    horloge_fenetre = depart_horloge
    perte_courante = float("nan")
    flops_crete = flops_crete_du_materiel()
    flops_par_token = cfg.model.flops_par_token()

    modele_calcul.train()
    for pas in range(pas_depart, t.max_steps):
        lr = taux_apprentissage(pas, cfg)
        for groupe in optimiseur.param_groups:
            groupe["lr"] = lr

        optimiseur.zero_grad(set_to_none=True)
        perte_cumulee = 0.0
        for micro in range(t.grad_accum):
            entree, cible = chargeur.lot(pas * t.grad_accum + micro)
            entree = entree.to(peripherique, non_blocking=True)
            cible = cible.to(peripherique, non_blocking=True)

            if distribue:
                # Ne synchroniser les gradients qu'au dernier micro-lot : sinon
                # on paie une réduction tout-à-tous à chaque accumulation.
                modele_calcul.require_backward_grad_sync = micro == t.grad_accum - 1

            with contexte:
                _, perte = modele_calcul(entree, cibles=cible)
                perte = perte / t.grad_accum
            scaler.scale(perte).backward()
            perte_cumulee += perte.item()

        if t.grad_clip > 0:
            scaler.unscale_(optimiseur)
            norme = torch.nn.utils.clip_grad_norm_(modele.parameters(), t.grad_clip)
            norme_grad = float(norme)
        else:
            norme_grad = float("nan")

        scaler.step(optimiseur)
        scaler.update()
        perte_courante = perte_cumulee

        if not math.isfinite(perte_courante):
            raise RuntimeError(
                f"La perte a divergé au pas {pas} (valeur : {perte_courante}). "
                f"Pistes : baisser train.lr, allonger train.warmup_steps, "
                f"ou vérifier les données."
            )

        # -- journal -------------------------------------------------------- #
        if maitre and (pas % t.log_every == 0 or pas == t.max_steps - 1):
            maintenant = time.perf_counter()
            ecoule = maintenant - horloge_fenetre
            horloge_fenetre = maintenant
            n_pas_fenetre = t.log_every if pas > pas_depart else 1
            tokens_par_s = tokens_par_pas * n_pas_fenetre / max(ecoule, 1e-9)
            mfu = (
                flops_par_token * tokens_par_s / flops_crete / monde
                if flops_crete > 0
                else 0.0
            )
            mesures = {
                "pas": pas,
                "perte": round(perte_courante, 5),
                "lr": lr,
                "norme_grad": round(norme_grad, 4),
                "tokens_vus": (pas + 1) * tokens_par_pas,
                "tokens_par_s": round(tokens_par_s, 1),
                "mfu": round(mfu, 4),
                "secondes": round(maintenant - depart_horloge, 2),
            }
            journal.ecrire(**mesures)
            ligne = (
                f"pas {pas:>6} · perte {perte_courante:.4f} · lr {lr:.2e} "
                f"· |g| {norme_grad:.2f} · {tokens_par_s:,.0f} tok/s".replace(",", " ")
            )
            if mfu > 0:
                ligne += f" · MFU {mfu * 100:.1f} %"
            dire(ligne)

        # -- validation ----------------------------------------------------- #
        if (
            chargeur_val is not None
            and maitre
            and t.eval_every > 0
            and (pas + 1) % t.eval_every == 0
        ):
            perte_val = evaluer(
                modele, chargeur_val, t.eval_batches, peripherique, contexte
            )
            journal.ecrire(pas=pas, perte_val=round(perte_val, 5))
            marque = ""
            if perte_val < meilleure_val:
                meilleure_val = perte_val
                sauvegarder(
                    dossier / "meilleur.pt", modele, optimiseur, cfg, pas + 1,
                    meilleure_val, scaler,
                )
                marque = "  ← meilleur"
            dire(
                f"       validation : perte {perte_val:.4f} "
                f"· perplexité {math.exp(min(perte_val, 20)):.1f}{marque}"
            )

        # -- sauvegarde ----------------------------------------------------- #
        if maitre and t.save_every > 0 and (pas + 1) % t.save_every == 0:
            sauvegarder(
                dossier / f"pas_{pas + 1:07d}.pt", modele, optimiseur, cfg,
                pas + 1, meilleure_val, scaler,
            )
            sauvegarder(
                dossier / "dernier.pt", modele, optimiseur, cfg, pas + 1,
                meilleure_val, scaler,
            )
            _nettoyer_checkpoints(dossier, t.garder_n_checkpoints)

    # -- fin ---------------------------------------------------------------- #
    if maitre:
        sauvegarder(
            dossier / "dernier.pt", modele, optimiseur, cfg, t.max_steps,
            meilleure_val, scaler,
        )
        journal.fermer()

    if distribue:
        import torch.distributed as dist

        dist.destroy_process_group()

    return ResultatEntrainement(
        pas_effectues=t.max_steps - pas_depart,
        perte_finale=perte_courante,
        meilleure_val=meilleure_val,
        tokens_vus=(t.max_steps - pas_depart) * tokens_par_pas,
        secondes=time.perf_counter() - depart_horloge,
        dossier=dossier,
    )
