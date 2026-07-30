"""Évaluation d'un modèle Futo.

Ce qu'on mesure, et pourquoi
----------------------------
**La perplexité** est la mesure de référence, mais elle a un défaut rédhibitoire
pour comparer deux modèles : elle dépend du tokenizer. Un modèle qui découpe le
français en gros tokens a mécaniquement une perplexité par token plus élevée
qu'un modèle qui le découpe finement, sans être moins bon pour autant. Comparer
deux perplexités calculées avec des vocabulaires différents ne veut rien dire.

**Les bits par octet** corrigent cela. On ramène la vraisemblance au texte
brut — le nombre d'octets UTF-8 — plutôt qu'au nombre de tokens. La mesure
devient indépendante du découpage, et donc comparable entre n'importe quels
modèles, y compris avec un compresseur classique. C'est la métrique à mettre en
avant dans une carte de modèle.

    bits/octet = (somme des log-vraisemblances négatives, en nats)
                 / (nombre d'octets UTF-8 × ln 2)

**Les sondes grammaticales** répondent à un problème pratique : un modèle de
120 à 400 millions de paramètres est trop petit pour les grands bancs d'essai de
connaissances, où il répondrait au niveau du hasard. Le mesurer là-dessus ne dit
rien. En revanche, la grammaire française — accords, subjonctif, ordre des
pronoms, élision — s'acquiert tôt et se mesure proprement par des paires
minimales : deux phrases quasi identiques dont une seule est correcte. On
demande au modèle laquelle il juge la plus probable. C'est la même idée que
BLiMP pour l'anglais, appliquée aux difficultés propres du français.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch

__all__ = [
    "ResultatPerplexite",
    "ResultatSondes",
    "ResultatSuggestions",
    "mesurer_perplexite",
    "mesurer_bits_par_octet",
    "mesurer_sondes",
    "mesurer_suggestions",
    "charger_sondes",
]


# --------------------------------------------------------------------------- #
# Perplexité et bits par octet
# --------------------------------------------------------------------------- #


@dataclass
class ResultatPerplexite:
    perte: float  # log-vraisemblance négative moyenne par token, en nats
    n_tokens: int
    n_octets: int | None = None

    @property
    def perplexite(self) -> float:
        # Au-delà de e^20 la valeur n'a plus de sens et déborde : on plafonne.
        return math.exp(min(self.perte, 20.0))

    @property
    def bits_par_token(self) -> float:
        return self.perte / math.log(2)

    @property
    def bits_par_octet(self) -> float | None:
        if not self.n_octets:
            return None
        return (self.perte * self.n_tokens) / (self.n_octets * math.log(2))

    def __str__(self) -> str:
        texte = (
            f"perte {self.perte:.4f} nats/token · "
            f"perplexité {self.perplexite:.2f} · "
            f"{self.bits_par_token:.3f} bit/token"
        )
        bpo = self.bits_par_octet
        if bpo is not None:
            texte += f" · {bpo:.3f} bit/octet"
        return texte


@torch.no_grad()
def mesurer_perplexite(modele, chargeur, n_lots: int, peripherique=None) -> ResultatPerplexite:
    """Perplexité sur les `n_lots` premiers lots du chargeur.

    Ces lots sont déterministes : deux appels donnent le même résultat, et deux
    modèles évalués avec le même chargeur voient exactement le même texte.
    """
    peripherique = peripherique or next(modele.parameters()).device
    etait = modele.training
    modele.eval()

    total_nll = 0.0
    total_tokens = 0
    for i in range(n_lots):
        entree, cible = chargeur.lot(i)
        entree, cible = entree.to(peripherique), cible.to(peripherique)
        _, perte = modele(entree, cibles=cible)
        n = cible.numel()
        total_nll += perte.item() * n
        total_tokens += n

    modele.train(etait)
    return ResultatPerplexite(perte=total_nll / max(total_tokens, 1), n_tokens=total_tokens)


@torch.no_grad()
def mesurer_bits_par_octet(
    modele, tokenizer, textes: list[str], block_size: int | None = None, peripherique=None
) -> ResultatPerplexite:
    """Bits par octet sur une liste de textes bruts.

    Chaque texte est encodé puis découpé en tranches de la longueur de contexte.
    Le tout premier token de chaque texte n'est pas prédit — il n'a aucun
    contexte — et il est donc exclu du décompte, sans quoi on pénaliserait le
    modèle pour une prédiction impossible.
    """
    peripherique = peripherique or next(modele.parameters()).device
    block_size = block_size or modele.cfg.block_size
    etait = modele.training
    modele.eval()

    total_nll = 0.0
    total_tokens = 0
    total_octets = 0

    for texte in textes:
        texte = unicodedata.normalize("NFC", texte)
        ids = tokenizer.encoder(texte)
        if len(ids) < 2:
            continue
        total_octets += len(texte.encode("utf-8"))

        # Découpe en tranches disjointes : chaque token est prédit une fois et
        # une seule, avec le contexte dont il dispose dans sa tranche.
        for debut in range(0, len(ids) - 1, block_size):
            tranche = ids[debut : debut + block_size + 1]
            if len(tranche) < 2:
                continue
            entree = torch.tensor([tranche[:-1]], dtype=torch.long, device=peripherique)
            cible = torch.tensor([tranche[1:]], dtype=torch.long, device=peripherique)
            _, perte = modele(entree, cibles=cible)
            n = cible.numel()
            total_nll += perte.item() * n
            total_tokens += n

    modele.train(etait)
    if total_tokens == 0:
        raise ValueError("Aucun token à évaluer : les textes fournis sont-ils vides ?")

    return ResultatPerplexite(
        perte=total_nll / total_tokens, n_tokens=total_tokens, n_octets=total_octets
    )


# --------------------------------------------------------------------------- #
# Suggestions : le bon mot est-il dans les k proposés ?
# --------------------------------------------------------------------------- #


@dataclass
class ResultatSuggestions:
    """Taux de réussite d'un clavier à suggestions.

    Deux mesures, et la seconde est la seule qui compte pour un clavier.

    Sur TOUS les tokens, on mesure la prédiction du morceau suivant, quel qu'il
    soit — y compris la fin d'un mot commencé (« aujourd' » puis « hui »). Ce
    chiffre est flatteur : compléter un mot déjà entamé est facile.

    Sur les DÉBUTS DE MOT seulement — les tokens qui commencent par une espace —
    on mesure ce que fait vraiment un clavier : proposer le mot suivant alors
    que l'utilisateur n'a encore rien tapé. C'est nettement plus dur, et c'est
    le chiffre à publier.
    """

    n_tokens: int = 0
    n_debuts_de_mot: int = 0
    reussites: dict[int, int] = field(default_factory=dict)
    reussites_mots: dict[int, int] = field(default_factory=dict)

    def taux(self, k: int) -> float:
        return self.reussites.get(k, 0) / max(self.n_tokens, 1)

    def taux_mots(self, k: int) -> float:
        return self.reussites_mots.get(k, 0) / max(self.n_debuts_de_mot, 1)

    def __str__(self) -> str:
        ks = sorted(self.reussites)
        tous = " · ".join(f"top-{k} {100 * self.taux(k):.1f} %" for k in ks)
        mots = " · ".join(f"top-{k} {100 * self.taux_mots(k):.1f} %" for k in ks)
        return (
            f"tous les tokens ({self.n_tokens}) : {tous}\n"
            f"  débuts de mot ({self.n_debuts_de_mot}) : {mots}"
        )


def _debuts_de_mot(tokenizer) -> torch.Tensor:
    """Masque booléen du vocabulaire : ce token ouvre-t-il un mot ?

    Un token ouvre un mot s'il se décode en une chaîne commençant par une
    espace. Les tokens spéciaux et la ponctuation collée n'en sont pas.
    """
    taille = tokenizer.vocab_size
    masque = torch.zeros(taille, dtype=torch.bool)
    for identifiant in range(taille):
        texte = tokenizer.decoder([identifiant], sauter_speciaux=False)
        masque[identifiant] = texte.startswith(" ") and len(texte) > 1
    return masque


@torch.no_grad()
def mesurer_suggestions(
    modele,
    chargeur,
    tokenizer,
    n_lots: int,
    ks: tuple[int, ...] = (1, 3, 4, 5),
    peripherique=None,
) -> ResultatSuggestions:
    """Le token attendu figure-t-il dans les k plus probables ?

    C'est la mesure directe d'un clavier à suggestions : quatre cases au-dessus
    des touches, le bon mot est-il dedans ? La perplexité ne répond pas à cette
    question — elle note la probabilité attribuée au bon mot, pas son rang.
    """
    peripherique = peripherique or next(modele.parameters()).device
    etait = modele.training
    modele.eval()

    masque_mots = _debuts_de_mot(tokenizer).to(peripherique)
    kmax = max(ks)
    resultat = ResultatSuggestions(reussites=dict.fromkeys(ks, 0),
                                   reussites_mots=dict.fromkeys(ks, 0))

    for i in range(n_lots):
        entree, cible = chargeur.lot(i)
        entree, cible = entree.to(peripherique), cible.to(peripherique)
        logits, _ = modele(entree, tous_les_pas=True)
        # rang de la cible : est-elle dans les kmax premiers ?
        meilleurs = logits.topk(kmax, dim=-1).indices          # (B, T, kmax)
        touche = meilleurs.eq(cible.unsqueeze(-1))             # (B, T, kmax)
        ouvre_un_mot = masque_mots[cible]                      # (B, T)

        resultat.n_tokens += cible.numel()
        resultat.n_debuts_de_mot += int(ouvre_un_mot.sum().item())
        for k in ks:
            dans_k = touche[..., :k].any(dim=-1)
            resultat.reussites[k] += int(dans_k.sum().item())
            resultat.reussites_mots[k] += int((dans_k & ouvre_un_mot).sum().item())

    modele.train(etait)
    return resultat


# --------------------------------------------------------------------------- #
# Sondes grammaticales
# --------------------------------------------------------------------------- #


@dataclass
class Paire:
    phenomene: str
    correct: str
    incorrect: str
    note: str = ""


@dataclass
class ResultatSondes:
    """Taux de réussite global et par phénomène.

    Deux façons de comparer les deux phrases d'une paire :

    * `justesse` — on compare les log-vraisemblances *totales*. C'est la
      convention de BLiMP, valable quand les deux phrases font la même longueur ;
    * `justesse_par_octet` — on divise chaque log-vraisemblance par le nombre
      d'octets de la phrase. Plus robuste quand les longueurs diffèrent (« L'homme »
      contre « Le homme »), car un texte plus long est mécaniquement moins probable.

    On rapporte les deux : si elles divergent beaucoup, c'est que le jeu de
    paires n'est pas assez équilibré en longueur, et il faut le dire.
    """

    n_paires: int
    n_justes: int
    n_justes_par_octet: int
    par_phenomene: dict[str, tuple[int, int]] = field(default_factory=dict)
    marge_moyenne: float = 0.0

    @property
    def justesse(self) -> float:
        return self.n_justes / max(self.n_paires, 1)

    @property
    def justesse_par_octet(self) -> float:
        return self.n_justes_par_octet / max(self.n_paires, 1)

    def __str__(self) -> str:
        lignes = [
            f"Sondes grammaticales : {self.n_justes}/{self.n_paires} "
            f"({self.justesse * 100:.1f} %) — par octet : "
            f"{self.justesse_par_octet * 100:.1f} % — hasard : 50,0 %",
        ]
        for phenomene, (justes, total) in sorted(self.par_phenomene.items()):
            barre = "█" * round(10 * justes / max(total, 1))
            lignes.append(
                f"  {phenomene:<34} {justes:>2}/{total:<2} "
                f"{justes / max(total, 1) * 100:>5.1f} %  {barre}"
            )
        return "\n".join(lignes)


def charger_sondes(chemin: str | Path) -> list[Paire]:
    """Lit le fichier JSONL des paires minimales."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(
            f"Jeu de sondes introuvable : {chemin}\n"
            f"Il est fourni avec le dépôt, dans data/sondes/."
        )
    paires: list[Paire] = []
    with chemin.open("r", encoding="utf-8") as fh:
        for numero, ligne in enumerate(fh, 1):
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#"):
                continue
            try:
                obj = json.loads(ligne)
            except json.JSONDecodeError as e:
                raise ValueError(f"{chemin}:{numero} : JSON invalide ({e}).") from None
            manquants = {"phenomene", "correct", "incorrect"} - set(obj)
            if manquants:
                raise ValueError(f"{chemin}:{numero} : champs manquants {sorted(manquants)}.")
            if obj["correct"] == obj["incorrect"]:
                raise ValueError(f"{chemin}:{numero} : les deux phrases sont identiques.")
            paires.append(
                Paire(
                    phenomene=obj["phenomene"],
                    correct=obj["correct"],
                    incorrect=obj["incorrect"],
                    note=obj.get("note", ""),
                )
            )
    if not paires:
        raise ValueError(f"{chemin} : aucune paire lisible.")
    return paires


@torch.no_grad()
def _log_vraisemblance(modele, tokenizer, phrase: str, peripherique) -> tuple[float, int]:
    """Log-vraisemblance totale d'une phrase, en nats, et sa taille en octets.

    La phrase est précédée du token de fin de document, qui joue le rôle de
    début de séquence : sans lui, le premier mot de la phrase serait prédit
    sans aucun contexte et sa vraisemblance ne voudrait rien dire.
    """
    phrase = unicodedata.normalize("NFC", phrase)
    ids = [tokenizer.id_fin] + tokenizer.encoder(phrase)
    if len(ids) > modele.cfg.block_size + 1:
        ids = ids[: modele.cfg.block_size + 1]

    entree = torch.tensor([ids[:-1]], dtype=torch.long, device=peripherique)
    cible = torch.tensor([ids[1:]], dtype=torch.long, device=peripherique)
    logits, _ = modele(entree, cibles=cible)
    log_probas = torch.log_softmax(logits.float(), dim=-1)
    retenus = log_probas.gather(2, cible.unsqueeze(-1)).squeeze(-1)
    return float(retenus.sum()), len(phrase.encode("utf-8"))


def mesurer_sondes(modele, tokenizer, chemin_ou_paires, peripherique=None) -> ResultatSondes:
    """Fait passer au modèle le jeu de paires minimales françaises."""
    peripherique = peripherique or next(modele.parameters()).device
    paires = (
        chemin_ou_paires
        if isinstance(chemin_ou_paires, list)
        else charger_sondes(chemin_ou_paires)
    )

    etait = modele.training
    modele.eval()

    n_justes = 0
    n_justes_octet = 0
    somme_marges = 0.0
    compte: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for paire in paires:
        lv_bon, oct_bon = _log_vraisemblance(modele, tokenizer, paire.correct, peripherique)
        lv_mauvais, oct_mauvais = _log_vraisemblance(
            modele, tokenizer, paire.incorrect, peripherique
        )

        juste = lv_bon > lv_mauvais
        juste_octet = (lv_bon / max(oct_bon, 1)) > (lv_mauvais / max(oct_mauvais, 1))

        n_justes += int(juste)
        n_justes_octet += int(juste_octet)
        somme_marges += lv_bon - lv_mauvais
        compte[paire.phenomene][0] += int(juste)
        compte[paire.phenomene][1] += 1

    modele.train(etait)
    return ResultatSondes(
        n_paires=len(paires),
        n_justes=n_justes,
        n_justes_par_octet=n_justes_octet,
        par_phenomene={k: (v[0], v[1]) for k, v in compte.items()},
        marge_moyenne=somme_marges / max(len(paires), 1),
    )
