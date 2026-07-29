"""Le tokenizer français de Futo.

C'est la pièce la plus spécifiquement française du projet. L'architecture d'un
décodeur n'a rien de national ; la façon de découper le français, si.

Choix retenus, et pourquoi
--------------------------
**BPE au niveau octet.** Le vocabulaire de départ est l'ensemble des 256 octets,
donc *aucun* texte n'est hors vocabulaire : pas de token « inconnu », pas de perte
sur un caractère rare, un emoji ou un idéogramme. C'est la garantie que le
tokenizer ne perd jamais rien, ce qui est vérifié par un test d'aller-retour.

**Une découpe préalable adaptée au français.** Les tokenizers courants sont
calibrés sur l'anglais, où la contraction se trouve à *droite* de l'apostrophe
(`do|n't`, `it|'s`). En français, l'élision est à *gauche* : `l'|homme`,
`qu'|il`, `aujourd'|hui`. Un motif anglophone découpe `l`, `'homme`, ce qui
sépare l'apostrophe de son proclitique et disperse le vocabulaire. Le motif
ci-dessous garde `l'` d'un bloc, et fait de même pour les clitiques accrochés par
un trait d'union (`dit-il`, `vas-y`, `y a-t-il`, `donne-le-moi`).

**Les deux apostrophes sont conservées.** Le français s'écrit tantôt avec
l'apostrophe droite `'` (U+0027), tantôt avec la typographique `’` (U+2019). On
pourrait tout ramener à l'une des deux, mais la normalisation est *destructrice* :
le modèle ne saurait plus restituer le texte d'origine. On garde donc les deux
formes ; le motif de découpe les traite à l'identique, et le coût se limite à
quelques dizaines de tokens dupliqués sur un vocabulaire de 32 768.

**Les chiffres sont coupés par groupes de trois au maximum.** `1 234` devient
`1` + ` 234`, jamais un token unique `1234`. Sans cela, le modèle apprend par
cœur des nombres fréquents et calcule très mal.

**Normalisation NFC.** `é` s'écrit soit en un point de code (U+00E9), soit en
deux (`e` + accent combinant). Sans normalisation, ce sont deux tokens
différents pour le même mot. NFC choisit la forme composée, celle que produisent
les claviers français. L'aller-retour est donc exact pour tout texte déjà en
NFC — ce que garantit le pipeline de données, qui normalise à l'entrée.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MOTIF_FRANCAIS",
    "TOKENS_SPECIAUX",
    "TokenizerFuto",
    "entrainer_tokenizer",
    "mesurer_fertilite",
]


# --------------------------------------------------------------------------- #
# Le motif de découpe préalable
# --------------------------------------------------------------------------- #

# Chaque alternative est commentée ; elles sont essayées dans l'ordre, la
# première qui correspond gagne. L'espace éventuel en tête (`[ ]?`) est attaché
# au morceau qui suit, convention du BPE au niveau octet : un mot en début de
# phrase et le même mot au milieu partagent alors le même token, à l'espace près.
_ALTERNATIVES: list[tuple[str, str]] = [
    (
        r"[ ]?(?i:aujourd|jusqu|lorsqu|puisqu|quoiqu|presqu|quelqu|entr)['’]",
        "élisions longues : aujourd', jusqu', lorsqu', puisqu', quoiqu', presqu'…",
    ),
    (
        r"[ ]?(?i:qu|[cdjlmnst])['’]",
        "élisions courtes : qu', c', d', j', l', m', n', s', t'",
    ),
    (
        r"-(?i:t[-'’])?(?i:je|tu|il|elle|on|nous|vous|ils|elles|moi|toi|lui|leur"
        r"|en|ci|là|le|la|les|y)(?![\p{L}])",
        "clitiques accrochés : dit-il, vas-y, y a-t-il, va-t'en, donne-le-moi, celui-ci",
    ),
    (r"[ ]?\p{L}+", "suites de lettres (accents et ligatures compris)"),
    (r"[ ]?\p{N}{1,3}", "chiffres, par groupes de trois au maximum"),
    (r"[ ]?[^\s\p{L}\p{N}]+[\r\n]*", "ponctuation et symboles, avec les retours à la ligne"),
    (r"\s*[\r\n]+", "sauts de ligne"),
    (r"\s+(?!\S)", "espaces en fin de texte"),
    (r"\s+", "toute autre suite d'espaces"),
]

MOTIF_FRANCAIS = "|".join(motif for motif, _ in _ALTERNATIVES)


def expliquer_motif() -> str:
    """Rend le motif lisible, une alternative par ligne (pour `futo tokenizer info`)."""
    largeur = max(len(m) for m, _ in _ALTERNATIVES)
    return "\n".join(f"  {m:<{largeur}}  # {c}" for m, c in _ALTERNATIVES)


# --------------------------------------------------------------------------- #
# Les tokens spéciaux
# --------------------------------------------------------------------------- #

# L'ordre fixe les identifiants : ne jamais insérer au milieu, seulement
# remplacer un slot réservé — sinon tous les checkpoints existants deviennent
# incompatibles avec le tokenizer.
TOKENS_SPECIAUX: list[str] = [
    "<|fin_de_texte|>",  # 0 — sépare les documents ; sert aussi de EOS
    "<|rembourrage|>",  # 1 — remplissage des lots (jamais appris : masqué dans la perte)
    "<|debut_de_texte|>",  # 2 — BOS, optionnel
]
# Douze emplacements réservés pour la suite (marqueurs de dialogue, rôles,
# appels d'outils…). Les réserver dès maintenant évite d'avoir à réentraîner le
# tokenizer, et donc le modèle, le jour où on en aura besoin.
TOKENS_SPECIAUX += [f"<|reserve_{i}|>" for i in range(12)]

ID_FIN_DE_TEXTE = 0
ID_REMBOURRAGE = 1
ID_DEBUT_DE_TEXTE = 2


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #


def _construire_vierge():
    """Assemble un tokenizer non entraîné, prêt à recevoir des fusions BPE."""
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(MOTIF_FRANCAIS), behavior="isolated", invert=False
            ),
            # `use_regex=False` : la découpe est déjà faite par le motif français
            # ci-dessus, ByteLevel ne doit pas réappliquer celle de GPT-2.
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def _lire_textes(chemins: Iterable[Path], octets_max: int | None = None) -> Iterator[str]:
    """Fournit les documents au trainer, sans tout charger en mémoire.

    Les fichiers sont lus par blocs de lignes ; `octets_max` permet de plafonner
    la quantité de texte utilisée (entraîner un BPE sur 2 Go suffit largement,
    au-delà on paie du temps sans gagner de qualité).
    """
    lus = 0
    tampon: list[str] = []
    for chemin in chemins:
        with Path(chemin).open("r", encoding="utf-8", errors="replace") as fh:
            for ligne in fh:
                tampon.append(ligne)
                lus += len(ligne.encode("utf-8"))
                if len(tampon) >= 10_000:
                    yield "".join(tampon)
                    tampon = []
                if octets_max is not None and lus >= octets_max:
                    if tampon:
                        yield "".join(tampon)
                    return
    if tampon:
        yield "".join(tampon)


def entrainer_tokenizer(
    fichiers: list[str | Path],
    sortie: str | Path,
    vocab_size: int = 32_768,
    frequence_min: int = 2,
    octets_max: int | None = None,
    verbeux: bool = True,
) -> TokenizerFuto:
    """Entraîne un BPE français et l'enregistre en un unique `tokenizer.json`.

    Aucun accès réseau n'est nécessaire : la bibliothèque `tokenizers` est une
    extension Rust installée depuis PyPI, elle n'a pas besoin du dépôt de modèles
    de Hugging Face pour *entraîner* un tokenizer.
    """
    from tokenizers import pre_tokenizers, trainers

    chemins = [Path(f) for f in fichiers]
    manquants = [str(c) for c in chemins if not c.exists()]
    if manquants:
        raise FileNotFoundError(f"Fichiers de corpus introuvables : {manquants}")

    if vocab_size <= len(TOKENS_SPECIAUX) + 256:
        raise ValueError(
            f"vocab_size ({vocab_size}) doit dépasser {len(TOKENS_SPECIAUX) + 256} "
            f"(les {len(TOKENS_SPECIAUX)} tokens spéciaux + les 256 octets de base)."
        )

    tok = _construire_vierge()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=frequence_min,
        special_tokens=TOKENS_SPECIAUX,
        # Sans cet alphabet initial, un octet absent du corpus d'entraînement
        # serait impossible à encoder plus tard : on impose les 256.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=verbeux,
    )
    tok.train_from_iterator(_lire_textes(chemins, octets_max), trainer=trainer)

    sortie = Path(sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(sortie))

    # Une fiche lisible à côté du tokenizer : on veut pouvoir répondre à
    # « d'où sort ce vocabulaire ? » six mois plus tard.
    fiche = {
        "vocab_size": tok.get_vocab_size(),
        "vocab_size_demande": vocab_size,
        "frequence_min": frequence_min,
        "octets_max": octets_max,
        "tokens_speciaux": TOKENS_SPECIAUX,
        "motif_decoupe": MOTIF_FRANCAIS,
        "normalisation": "NFC",
        "corpus": [str(c) for c in chemins],
    }
    sortie.with_suffix(".fiche.json").write_text(
        json.dumps(fiche, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return TokenizerFuto(sortie)


# --------------------------------------------------------------------------- #
# Utilisation
# --------------------------------------------------------------------------- #


class TokenizerFuto:
    """Enveloppe mince autour du tokenizer entraîné.

    N'ajoute rien de magique : elle expose l'encodage, le décodage, les
    identifiants spéciaux, et les mesures de qualité utilisées dans les tests.
    """

    def __init__(self, chemin: str | Path) -> None:
        from tokenizers import Tokenizer

        chemin = Path(chemin)
        if not chemin.exists():
            raise FileNotFoundError(
                f"Tokenizer introuvable : {chemin}\n"
                f"Entraînez-en un avec : futo tokenizer entrainer"
            )
        self.chemin = chemin
        self._tok = Tokenizer.from_file(str(chemin))
        self.id_fin = self._id_special("<|fin_de_texte|>", ID_FIN_DE_TEXTE)
        self.id_rembourrage = self._id_special("<|rembourrage|>", ID_REMBOURRAGE)
        self.id_debut = self._id_special("<|debut_de_texte|>", ID_DEBUT_DE_TEXTE)

    def _id_special(self, jeton: str, defaut: int) -> int:
        trouve = self._tok.token_to_id(jeton)
        return defaut if trouve is None else trouve

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def dtype_shards(self):
        """Le plus petit entier non signé capable de porter ce vocabulaire.

        uint16 tient jusqu'à 65 535 : à 32 768 tokens, les shards pèsent deux
        octets par token au lieu de quatre. Sur 10 milliards de tokens, cela fait
        20 Go au lieu de 40.
        """
        import numpy as np

        return np.uint16 if self.vocab_size <= 65_535 else np.uint32

    def encoder(self, texte: str, ajouter_fin: bool = False) -> list[int]:
        ids = self._tok.encode(texte, add_special_tokens=False).ids
        if ajouter_fin:
            ids.append(self.id_fin)
        return ids

    def encoder_lot(self, textes: list[str], ajouter_fin: bool = False) -> list[list[int]]:
        lots = self._tok.encode_batch(textes, add_special_tokens=False)
        if ajouter_fin:
            return [e.ids + [self.id_fin] for e in lots]
        return [e.ids for e in lots]

    def decoder(self, ids: list[int], sauter_speciaux: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=sauter_speciaux)

    def __len__(self) -> int:
        return self.vocab_size


# --------------------------------------------------------------------------- #
# Mesures de qualité
# --------------------------------------------------------------------------- #


@dataclass
class Fertilite:
    """Ce qu'on mesure pour dire si un tokenizer est bon sur du français.

    - `tokens_par_mot` : le nombre moyen de tokens par mot. C'est la mesure qui
      compte : à contexte égal, un tokenizer à 1,4 token/mot fait tenir 30 % de
      texte en plus qu'un tokenizer à 1,8, et l'entraînement coûte d'autant moins.
    - `octets_par_token` : l'autre face de la même pièce, indépendante du
      découpage en mots — utile pour comparer des langues entre elles.
    - `mots_en_un_token` : la part des mots rendus par un seul token. Un
      tokenizer anglophone appliqué au français fait chuter cette part, parce que
      les terminaisons françaises (-ement, -tion, -aient) n'ont pas de token.
    """

    n_mots: int
    n_tokens: int
    n_octets: int
    mots_en_un_token: float

    @property
    def tokens_par_mot(self) -> float:
        return self.n_tokens / max(self.n_mots, 1)

    @property
    def octets_par_token(self) -> float:
        return self.n_octets / max(self.n_tokens, 1)

    def __str__(self) -> str:
        return (
            f"{self.tokens_par_mot:.3f} token/mot · "
            f"{self.octets_par_token:.2f} octet/token · "
            f"{self.mots_en_un_token * 100:.1f} % des mots en un seul token "
            f"({self.n_mots} mots)"
        )


def mesurer_fertilite(tokenizer: TokenizerFuto, texte: str) -> Fertilite:
    """Calcule la fertilité sur un texte donné.

    Le découpage en mots se fait sur les espaces après normalisation NFC : c'est
    grossier, mais c'est la même règle pour tous les tokenizers comparés, donc la
    comparaison reste juste.
    """
    texte = unicodedata.normalize("NFC", texte)
    mots = [m for m in texte.split() if m]
    ids = tokenizer.encoder(texte)

    # Nombre de mots rendus par un seul token : on ré-encode mot à mot, précédé
    # d'une espace, comme le mot apparaîtrait au fil du texte.
    seuls = sum(1 for m in mots if len(tokenizer.encoder(" " + m)) == 1)

    return Fertilite(
        n_mots=len(mots),
        n_tokens=len(ids),
        n_octets=len(texte.encode("utf-8")),
        mots_en_un_token=seuls / max(len(mots), 1),
    )
