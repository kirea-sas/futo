"""Le transformeur décodeur de Futo.

Une implémentation « à la main », lisible et sans dépendance autre que PyTorch :
on veut pouvoir lire et modifier chaque ligne du moteur. L'architecture reprend
ce qui s'est stabilisé depuis Llama : pré-normalisation RMSNorm, positions
rotatives (RoPE), attention à requêtes groupées (GQA), MLP SwiGLU, aucun biais,
poids d'entrée et de sortie liés.

Pièges traités ici, et pourquoi ils comptent
--------------------------------------------
1. **RoPE se calcule en float32.** En bf16, `1 / 10000^(2i/d)` perd assez de
   précision pour que deux positions éloignées finissent avec le même angle.
   On calcule cos/sin en float32 puis on les applique avant tout autocast.
2. **`is_causal=True` n'est valable que si q_len == kv_len.** Le masque de
   `scaled_dot_product_attention` est aligné en haut à gauche : avec un cache KV
   et une seule requête, `is_causal=True` masquerait *tout* sauf le premier
   token. Voir `_masque_attention`.
3. **RMSNorm normalise en float32.** La somme des carrés déborde vite en fp16 et
   perd sa précision en bf16 ; on remonte en float32 pour la statistique, puis on
   redescend.
4. **Les projections de sortie sont initialisées plus petit.** Sans le facteur
   1/√(2·n_layer), la variance des activations croît avec la profondeur et les
   premiers pas divergent.
5. **Le decay ne s'applique qu'aux matrices.** Régulariser les gains de
   RMSNorm et les embeddings dégrade le modèle : voir `groupes_parametres`.
6. **Les poids liés doivent l'être *avant* de compter les paramètres**, sinon on
   surestime la taille du modèle de V·d.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

__all__ = ["Futo", "RMSNorm", "Attention", "MLP", "Bloc", "CacheKV"]


# --------------------------------------------------------------------------- #
# Briques
# --------------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    """Normalisation par la racine du carré moyen (Zhang & Sennrich, 2019).

    Moins chère que LayerNorm — pas de moyenne à retrancher, pas de biais — et
    empiriquement aussi stable pour les décodeurs.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Piège n° 3 : la statistique se calcule en float32, quel que soit le
        # dtype d'entrée, puis on revient au dtype d'origine.
        dtype_entree = x.dtype
        x32 = x.float()
        normalise = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normalise.to(dtype_entree)) * self.weight

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}"


def construire_rope(
    head_dim: int,
    longueur: int,
    theta: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Précalcule les cosinus et sinus de RoPE (Su et al., 2021).

    Renvoie deux tenseurs de forme (longueur, head_dim / 2), en float32.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim doit être pair pour RoPE, reçu {head_dim}.")
    # Piège n° 1 : tout en float32.
    frequences = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(longueur, dtype=torch.float32, device=device)
    angles = torch.outer(positions, frequences)  # (longueur, head_dim/2)
    return angles.cos(), angles.sin()


def appliquer_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Fait tourner les paires de dimensions de `x` selon leur position.

    `x` a la forme (B, n_tetes, T, head_dim) ; `cos` et `sin` la forme
    (T, head_dim/2). On sépare les dimensions paires et impaires — convention
    « moitié/moitié » de Llama plutôt que l'entrelacement de l'article original,
    les deux sont équivalents à une permutation près du vocabulaire de poids.
    """
    x1, x2 = x.float().chunk(2, dim=-1)  # deux fois (B, H, T, head_dim/2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    tourne = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return tourne.to(x.dtype)


@dataclass
class CacheKV:
    """Cache clé/valeur pour la génération incrémentale.

    Les tenseurs ont la forme (B, n_kv_head, T_vu, head_dim). Ils grandissent par
    concaténation : simple, et suffisant tant qu'on génère quelques centaines de
    tokens. Pour de la production, on préallouerait un tampon de taille fixe.
    """

    cles: list[torch.Tensor | None]
    valeurs: list[torch.Tensor | None]

    @classmethod
    def vide(cls, n_layer: int) -> CacheKV:
        return cls(cles=[None] * n_layer, valeurs=[None] * n_layer)

    @property
    def longueur(self) -> int:
        premiere = self.cles[0]
        return 0 if premiere is None else premiere.shape[-2]


class Attention(nn.Module):
    """Attention causale multi-têtes à requêtes groupées (GQA).

    GQA (Ainslie et al., 2023) partage une tête clé/valeur entre plusieurs têtes
    de requête. À qualité quasi identique, le cache KV de l'inférence est divisé
    par `n_head / n_kv_head` — c'est ce qui rend la génération tenable en mémoire.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.dropout = cfg.dropout

        d = cfg.d_model
        self.q_proj = nn.Linear(d, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, d, bias=False)
        self.o_proj._est_projection_de_sortie = True  # repéré par l'initialisation

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: CacheKV | None = None,
        indice_couche: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE s'applique aux requêtes et aux clés, jamais aux valeurs.
        q = appliquer_rope(q, cos, sin)
        k = appliquer_rope(k, cos, sin)

        if cache is not None:
            k_ancien = cache.cles[indice_couche]
            v_ancien = cache.valeurs[indice_couche]
            if k_ancien is not None and v_ancien is not None:
                k = torch.cat([k_ancien, k], dim=2)
                v = torch.cat([v_ancien, v], dim=2)
            cache.cles[indice_couche] = k
            cache.valeurs[indice_couche] = v

        # GQA : PyTorch ≥ 2.5 sait répéter les têtes kv lui-même (enable_gqa),
        # ce qui évite de matérialiser un tenseur n_rep fois plus gros.
        if self.n_rep > 1 and not _SDPA_SUPPORTE_GQA:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        kwargs = {}
        if self.n_rep > 1 and _SDPA_SUPPORTE_GQA:
            kwargs["enable_gqa"] = True

        est_causal, masque = _masque_attention(T, k.shape[-2], x.device)
        sortie = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=masque,
            is_causal=est_causal,
            dropout_p=self.dropout if self.training else 0.0,
            **kwargs,
        )
        sortie = sortie.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(sortie)


# `enable_gqa` est apparu dans PyTorch 2.5 ; on teste une fois au chargement.
def _detecter_gqa() -> bool:
    try:
        q = torch.zeros(1, 2, 2, 2)
        kv = torch.zeros(1, 1, 2, 2)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
    except (TypeError, RuntimeError):
        return False
    return True


_SDPA_SUPPORTE_GQA = _detecter_gqa()


def _masque_attention(
    q_len: int, kv_len: int, device: torch.device
) -> tuple[bool, torch.Tensor | None]:
    """Choisit le masque causal correct selon la situation.

    Piège n° 2. Trois cas, et un seul est géré par `is_causal=True` :

    * **entraînement / préremplissage complet** (q_len == kv_len) : le masque
      triangulaire standard, `is_causal=True` fait exactement ce qu'il faut ;
    * **décodage token par token** (q_len == 1) : la requête est la dernière
      position, elle a le droit de voir tout le cache — aucun masque ;
    * **préremplissage par morceaux** (1 < q_len < kv_len) : les q_len requêtes
      occupent les dernières positions ; il faut un masque explicitement décalé,
      car `is_causal` l'alignerait en haut à gauche et masquerait le cache.
    """
    if q_len == kv_len:
        return True, None
    if q_len == 1:
        return False, None
    decalage = kv_len - q_len
    lignes = torch.arange(q_len, device=device).unsqueeze(1) + decalage
    colonnes = torch.arange(kv_len, device=device).unsqueeze(0)
    return False, (colonnes <= lignes).unsqueeze(0).unsqueeze(0)


class MLP(nn.Module):
    """Réseau à propagation avant SwiGLU (Shazeer, 2020).

    `down(silu(gate(x)) * up(x))` — trois matrices au lieu de deux, mais une
    dimension cachée réduite d'un tiers pour compenser. Gain net constaté sur
    la perplexité à budget de paramètres égal.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_ff is not None
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.down_proj._est_projection_de_sortie = True
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Bloc(nn.Module):
    """Un bloc du décodeur, en pré-normalisation.

    La pré-normalisation (norme *avant* le sous-module, résiduel non normalisé)
    laisse un chemin résiduel propre de l'entrée à la sortie : c'est ce qui
    permet d'entraîner des dizaines de couches sans réglage fin du warmup.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norme_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norme_mlp = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: CacheKV | None = None,
        indice_couche: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.norme_attn(x), cos, sin, cache, indice_couche)
        x = x + self.mlp(self.norme_mlp(x))
        return x


# --------------------------------------------------------------------------- #
# Le modèle
# --------------------------------------------------------------------------- #


class Futo(nn.Module):
    """Modèle de langue causal.

    Exemple minimal ::

        cfg = ModelConfig(vocab_size=1000, n_layer=2, n_head=4, n_kv_head=2, d_model=128)
        modele = Futo(cfg)
        logits, perte = modele(entree, cibles=cible)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.blocs = nn.ModuleList([Bloc(cfg) for _ in range(cfg.n_layer)])
        self.norme_finale = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.tete = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Piège n° 6 : lier les poids AVANT toute initialisation ou comptage.
        if cfg.tie_weights:
            self.tete.weight = self.embeddings.weight

        self.apply(self._initialiser)
        # Piège n° 4 : les projections qui écrivent dans le résiduel démarrent
        # plus petit, sinon la variance s'accumule couche après couche.
        echelle = cfg.init_std / math.sqrt(2 * cfg.n_layer)
        for module in self.modules():
            if getattr(module, "_est_projection_de_sortie", False):
                nn.init.normal_(module.weight, mean=0.0, std=echelle)

        # Les tables cos/sin ne sont pas des paramètres : `persistent=False` les
        # exclut du state_dict, ce qui permet de changer block_size sans casser
        # la compatibilité des checkpoints.
        cos, sin = construire_rope(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # -- initialisation ---------------------------------------------------- #

    def _initialiser(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    # -- passe avant -------------------------------------------------------- #

    def forward(
        self,
        entree: torch.Tensor,
        cibles: torch.Tensor | None = None,
        cache: CacheKV | None = None,
        indice_ignore: int = -100,
        tous_les_pas: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Calcule les logits, et la perte si `cibles` est fourni.

        `entree` et `cibles` ont la forme (B, T). Les positions de `cibles`
        valant `indice_ignore` ne comptent pas dans la perte : c'est ainsi qu'on
        neutralise le rembourrage et, si on le souhaite, le premier token de
        chaque document.

        Sans `cibles`, seul le dernier pas est projeté sur le vocabulaire : c'est
        tout ce dont la génération a besoin, et cela évite une multiplication
        (T−1)×V inutile. `tous_les_pas=True` force le calcul complet, nécessaire
        pour noter une séquence entière sans fournir de cibles.
        """
        B, T = entree.shape
        decalage = cache.longueur if cache is not None else 0
        if decalage + T > self.cfg.block_size:
            raise ValueError(
                f"Séquence trop longue : {decalage} tokens en cache + {T} nouveaux "
                f"dépassent le contexte de {self.cfg.block_size}."
            )

        cos = self.rope_cos[decalage : decalage + T]
        sin = self.rope_sin[decalage : decalage + T]

        x = self.dropout(self.embeddings(entree))
        for i, bloc in enumerate(self.blocs):
            if self.cfg.dropout == 0 and self.training and self._checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    bloc, x, cos, sin, cache, i, use_reentrant=False
                )
            else:
                x = bloc(x, cos, sin, cache, i)
        x = self.norme_finale(x)

        if cibles is None:
            logits = self.tete(x if tous_les_pas else x[:, -1:, :])
            return logits, None

        logits = self.tete(x)
        perte = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            cibles.reshape(-1),
            ignore_index=indice_ignore,
        )
        return logits, perte

    _checkpointing: bool = False

    def activer_gradient_checkpointing(self, actif: bool = True) -> None:
        """Recalcule les activations à l'arrière au lieu de les stocker.

        Divise environ par 3 la mémoire d'activation, contre ~30 % de temps de
        calcul en plus. À n'activer que si le lot ne tient pas en mémoire.
        """
        self._checkpointing = actif

    # -- inférence ---------------------------------------------------------- #

    @torch.no_grad()
    def generer(
        self,
        amorce: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        token_fin: int | None = None,
        generateur: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Complète `amorce` (B, T) et renvoie la séquence entière (B, T + n).

        `temperature=0` bascule en décodage glouton (argmax). `top_k` et `top_p`
        se combinent : on garde l'intersection des deux filtres.
        """
        etait_en_entrainement = self.training
        self.eval()
        try:
            cache = CacheKV.vide(self.cfg.n_layer)
            sequence = amorce
            # Préremplissage : une seule passe sur toute l'amorce.
            logits, _ = self(amorce, cache=cache)

            fini = torch.zeros(amorce.shape[0], dtype=torch.bool, device=amorce.device)
            for _ in range(max_tokens):
                logits_derniers = logits[:, -1, :].float()
                suivant = _echantillonner(
                    logits_derniers, temperature, top_k, top_p, generateur
                )
                if token_fin is not None:
                    # Une fois qu'une séquence a produit le token de fin, on la
                    # fige : elle ne doit plus influencer le résultat.
                    suivant = torch.where(
                        fini.unsqueeze(1), torch.full_like(suivant, token_fin), suivant
                    )
                    fini |= suivant.squeeze(1) == token_fin
                sequence = torch.cat([sequence, suivant], dim=1)
                if token_fin is not None and bool(fini.all()):
                    break
                if sequence.shape[1] >= self.cfg.block_size:
                    break
                logits, _ = self(suivant, cache=cache)
            return sequence
        finally:
            self.train(etait_en_entrainement)

    # -- utilitaires -------------------------------------------------------- #

    def nombre_parametres(self, avec_embeddings: bool = True) -> int:
        """Compte les paramètres réellement alloués (poids liés comptés une fois)."""
        vus: set[int] = set()
        total = 0
        for nom, p in self.named_parameters():
            if id(p) in vus:
                continue
            vus.add(id(p))
            if not avec_embeddings and ("embeddings" in nom or nom.startswith("tete")):
                continue
            total += p.numel()
        return total

    def groupes_parametres(self, weight_decay: float) -> list[dict]:
        """Sépare les paramètres régularisés des autres.

        Piège n° 5 : on ne régularise que les tenseurs de dimension ≥ 2, c'est-à-dire
        les matrices de projection et les embeddings. Les gains de RMSNorm sont de
        dimension 1 : leur appliquer un decay les tire vers zéro et éteint
        progressivement les couches.
        """
        vus: set[int] = set()
        avec, sans = [], []
        for p in self.parameters():
            if not p.requires_grad or id(p) in vus:
                continue
            vus.add(id(p))
            (avec if p.dim() >= 2 else sans).append(p)
        return [
            {"params": avec, "weight_decay": weight_decay},
            {"params": sans, "weight_decay": 0.0},
        ]

    def redimensionner_contexte(self, block_size: int) -> None:
        """Étend (ou réduit) la longueur de contexte en recalculant RoPE.

        Utile pour évaluer sur des séquences plus longues que celles vues à
        l'entraînement — les résultats se dégradent vite au-delà, RoPE
        n'extrapolant pas de lui-même.
        """
        self.cfg.block_size = block_size
        cos, sin = construire_rope(
            self.cfg.head_dim, block_size, self.cfg.rope_theta, self.rope_cos.device
        )
        self.register_buffer("rope_cos", cos.to(self.rope_cos.dtype), persistent=False)
        self.register_buffer("rope_sin", sin.to(self.rope_sin.dtype), persistent=False)


def _echantillonner(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generateur: torch.Generator | None,
) -> torch.Tensor:
    """Tire un token par ligne de `logits` (B, V). Renvoie (B, 1)."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and 0 < top_k < logits.size(-1):
        seuil = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < seuil, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        tries, indices = torch.sort(logits, descending=True, dim=-1)
        cumul = torch.softmax(tries, dim=-1).cumsum(dim=-1)
        # On retire ce qui dépasse le seuil, en gardant toujours le premier
        # token : sinon un pic de probabilité > top_p viderait la distribution.
        a_retirer = cumul - torch.softmax(tries, dim=-1) > top_p
        a_retirer[:, 0] = False
        tries = tries.masked_fill(a_retirer, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(1, indices, tries)

    probabilites = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilites, num_samples=1, generator=generateur)
