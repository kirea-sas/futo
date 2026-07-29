"""Interface en ligne de commande de Futo.

Argparse plutôt qu'une bibliothèque tierce : la CLI est le point d'entrée du
projet, il serait dommage qu'elle impose une dépendance de plus. Toutes les
commandes acceptent `--set chemin.cle=valeur` pour surcharger la configuration
sans toucher au YAML — pratique pour balayer un hyperparamètre depuis un script.
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

__all__ = ["main", "construire_analyseur"]

RACINE = Path(__file__).resolve().parent.parent
CORPUS_EXEMPLE = "data/echantillon/*.txt"


def _etendre(motifs: list[str]) -> list[Path]:
    """Développe les jokers, en gardant l'ordre et sans doublon.

    Le shell le fait déjà sous Unix, mais pas sous Windows, et pas quand le motif
    est passé entre guillemets. On le refait donc nous-mêmes.
    """
    vus: dict[str, None] = {}
    for motif in motifs:
        trouves = sorted(glob.glob(motif)) or ([motif] if Path(motif).exists() else [])
        if not trouves:
            raise FileNotFoundError(f"Aucun fichier ne correspond à « {motif} ».")
        for t in trouves:
            vus.setdefault(t, None)
    return [Path(v) for v in vus]


def _milliers(n: float) -> str:
    return f"{n:,.0f}".replace(",", " ")


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #


def cmd_info(args) -> int:
    """Décrit une configuration sans rien entraîner : taille, mémoire, budget."""
    from .config import charger_config

    cfg = charger_config(args.config, args.set)
    m = cfg.model

    print(cfg.resume())
    print()

    n = m.nombre_parametres()
    # Mémoire de l'optimiseur : les poids en fp32 (4 o), plus les deux moments
    # d'Adam (4 o chacun), plus les gradients (4 o) — soit 16 octets par
    # paramètre. Les activations s'y ajoutent et dépendent du lot.
    octets_etat = n * 16
    print("Empreinte mémoire (hors activations) :")
    print(f"  poids + gradients + moments d'Adam : {octets_etat / 2**30:.2f} Gio")
    print(f"  poids seuls en bf16                : {n * 2 / 2**30:.2f} Gio")
    print()

    flops_token = m.flops_par_token()
    total_tokens = cfg.train.max_steps * cfg.tokens_par_pas()
    flops_total = flops_token * total_tokens
    print("Coût de calcul :")
    print(f"  {flops_token / 1e9:.2f} GFLOP par token (avant + arrière)")
    print(f"  {flops_total / 1e18:.3f} EFLOP pour l'entraînement complet")
    print()

    from .train import (
        FLOPS_CRETE,
        choisir_peripherique,
        flops_crete_du_materiel,
        nom_du_materiel,
    )

    def duree(heures: float) -> str:
        if heures < 0.05:
            return f"{heures * 60:.0f} min"
        if heures < 72:
            return f"{heures:.1f} h"
        return f"{heures / 24:.1f} j"

    # D'abord la machine sur laquelle la commande tourne : c'est le chiffre que
    # l'utilisateur cherche en premier.
    peripherique = choisir_peripherique()
    local = nom_du_materiel(peripherique)
    crete_locale = flops_crete_du_materiel(local)
    # 18 % pour Apple : mesuré à 19,1 % sur un M2 Max avec futo-mac en bf16,
    # on garde une marge. C'est UNE mesure sur UNE configuration — d'où
    # « futo bench », qui mesure la vôtre au lieu de l'extrapoler.
    mfu_local = 0.40 if peripherique.type == "cuda" else 0.18
    print(f"Votre machine : {local} ({peripherique.type})")
    if crete_locale > 0:
        print(
            f"  environ {duree(flops_total / (crete_locale * mfu_local) / 3600)} "
            f"à {mfu_local * 100:.0f} % de MFU"
        )
    elif peripherique.type == "cpu":
        print("  entraînement sur processeur : utilisable pour futo-tiny seulement.")
    else:
        print("  matériel non répertorié : durée inestimable, lancez et mesurez.")
    print()

    # Puis le reste, séparé en deux familles : le MFU atteignable n'est pas du
    # tout le même sur CUDA et sur le GPU intégré d'un Mac.
    nvidia = ["H100", "A100", "L40S", "4090", "3090"]
    apple = ["M3 Ultra", "M4 Max", "M3 Max", "M2 Max", "M4 Pro", "M2"]

    print("Ailleurs, à 40 % de MFU (cartes NVIDIA) :")
    for nom in nvidia:
        print(f"  {nom:<9} {duree(flops_total / (FLOPS_CRETE[nom] * 0.40) / 3600):>9}")
    print()
    print("À 18 % de MFU (Apple Silicon, mesuré sur M2 Max — voir docs/MAC.md) :")
    for nom in apple:
        print(f"  {nom:<9} {duree(flops_total / (FLOPS_CRETE[nom] * 0.18) / 3600):>9}")
    print()
    print("  Ces durées sont CALCULÉES, pas mesurées. 40 % de MFU est atteignable")
    print("  mais optimiste : sous 200 M de paramètres, le surcoût des noyaux et")
    print("  du chargement fait souvent tomber le MFU à 20-30 % sur GPU. Le")
    print("  chiffre Apple s'appuie sur une seule mesure (M2 Max, futo-mac, bf16).")
    print("  Pour un chiffre MESURÉ sur cette machine : futo bench <config>")
    return 0


def cmd_bench(args) -> int:
    """Mesure le débit réel de la configuration sur cette machine."""
    from .config import charger_config
    from .train import mesurer_debit

    cfg = charger_config(args.config, args.set)
    print(cfg.resume())
    print()

    resultat = mesurer_debit(
        cfg, micro_lots=args.micro_lots, echauffement=args.echauffement,
        verbeux=not args.silencieux,
    )

    print()
    print(f"Mesuré sur {resultat.materiel} en {resultat.dtype} :")
    print(f"  {_milliers(resultat.tokens_par_s)} tokens/s")
    if resultat.mfu > 0:
        print(f"  MFU {resultat.mfu * 100:.1f} % (rapporté à une crête estimée)")
    print(f"  {resultat.secondes_par_pas:.2f} s par pas d'optimisation "
          f"({_milliers(cfg.tokens_par_pas())} tokens)")
    print()

    tokens_totaux = cfg.train.max_steps * cfg.tokens_par_pas()
    secondes = resultat.duree_estimee(tokens_totaux)
    heures = secondes / 3600
    duree = f"{heures:.1f} h" if heures < 72 else f"{heures / 24:.1f} jours"
    print(f"Entraînement complet de « {cfg.nom} » à ce débit :")
    print(f"  {cfg.train.max_steps} pas · {_milliers(tokens_totaux)} tokens · {duree}")
    print()
    print("  Ce chiffre-ci est MESURÉ, pas calculé : c'est celui sur lequel")
    print("  décider. Il suppose un débit constant, ce qui est optimiste sur une")
    print("  machine de bureau — veille, thermique, autres applications.")
    return 0


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #


def cmd_tokenizer_entrainer(args) -> int:
    from .tokenizer import entrainer_tokenizer, mesurer_fertilite

    fichiers = _etendre(args.corpus)
    octets = sum(f.stat().st_size for f in fichiers)
    print(f"Entraînement du tokenizer sur {len(fichiers)} fichier(s), "
          f"{_milliers(octets)} octets.")
    if octets < 1_000_000:
        print("  Attention : moins d'un mégaoctet de texte. Le vocabulaire obtenu")
        print("  sera pauvre. C'est acceptable pour un test, pas pour un vrai modèle.")

    tok = entrainer_tokenizer(
        fichiers,
        args.sortie,
        vocab_size=args.vocab,
        frequence_min=args.frequence_min,
        octets_max=args.octets_max,
        verbeux=not args.silencieux,
    )
    print(f"\nTokenizer enregistré : {args.sortie}")
    print(f"  vocabulaire obtenu : {tok.vocab_size} "
          f"(demandé : {args.vocab})")

    echantillon = fichiers[0].read_text(encoding="utf-8", errors="replace")[:200_000]
    print(f"  fertilité mesurée  : {mesurer_fertilite(tok, echantillon)}")

    exemple = "L'homme qu'elle avait rencontré aujourd'hui n'était pas celui-ci."
    ids = tok.encoder(exemple)
    print(f"\n  Exemple : {exemple}")
    print(f"  {len(ids)} tokens : {[tok.decoder([i], sauter_speciaux=False) for i in ids]}")
    return 0


def cmd_tokenizer_info(args) -> int:
    from .tokenizer import TOKENS_SPECIAUX, TokenizerFuto, expliquer_motif, mesurer_fertilite

    tok = TokenizerFuto(args.tokenizer)
    print(f"Tokenizer : {args.tokenizer}")
    print(f"  vocabulaire : {tok.vocab_size}")
    print(f"  type des shards : {tok.dtype_shards.__name__}")
    print(f"  tokens spéciaux : {len(TOKENS_SPECIAUX)}")
    for i, jeton in enumerate(TOKENS_SPECIAUX[:3]):
        print(f"    {i:>2}  {jeton}")
    print(f"    …   {len(TOKENS_SPECIAUX) - 3} emplacements réservés")
    print("\nMotif de découpe préalable (adapté au français) :")
    print(expliquer_motif())

    if args.texte:
        ids = tok.encoder(args.texte)
        print(f"\nTexte : {args.texte}")
        print(f"  {len(ids)} tokens")
        print(f"  {[tok.decoder([i], sauter_speciaux=False) for i in ids]}")
        print(f"  aller-retour exact : {tok.decoder(ids) == args.texte}")
    if args.corpus:
        for f in _etendre(args.corpus):
            texte = f.read_text(encoding="utf-8", errors="replace")
            print(f"\n{f.name} : {mesurer_fertilite(tok, texte)}")
    return 0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def cmd_data_preparer(args) -> int:
    from .data import preparer_corpus
    from .tokenizer import TokenizerFuto

    fichiers = _etendre(args.corpus)
    tok = TokenizerFuto(args.tokenizer)
    print(f"Préparation de {len(fichiers)} fichier(s) avec un vocabulaire de "
          f"{tok.vocab_size} tokens.")

    resultat = preparer_corpus(
        fichiers,
        tok,
        args.sortie,
        tokens_par_shard=args.tokens_par_shard,
        fraction_val=args.fraction_val,
        separateur=args.separateur,
        journal=print if not args.silencieux else None,
    )
    print(f"\n{resultat}")
    print(f"Shards écrits dans {args.sortie}")
    return 0


def cmd_data_info(args) -> int:
    from .data import lire_entete

    chemins = sorted(Path(args.dossier).glob("*.bin"))
    if not chemins:
        print(f"Aucun shard dans {args.dossier}.", file=sys.stderr)
        return 1
    total = 0
    for chemin in chemins:
        entete = lire_entete(chemin)
        total += entete.n_tokens
        print(f"  {chemin.name:<24} {_milliers(entete.n_tokens):>15} tokens "
              f"· {entete.dtype} · vocab {entete.vocab_size}")
    print(f"  {'TOTAL':<24} {_milliers(total):>15} tokens")
    return 0


def cmd_data_telecharger(args) -> int:
    """Explique comment récupérer un vrai corpus français.

    Volontairement non automatisé : télécharger plusieurs dizaines de gigaoctets
    engage l'utilisateur, et les identifiants exacts des jeux de données changent.
    Mieux vaut une marche à suivre honnête qu'un script qui échoue en silence.
    """
    print(RECETTE_CORPUS)
    return 0


RECETTE_CORPUS = """\
Récupérer un vrai corpus français
=================================

Le corpus d'exemple livré avec le dépôt (data/echantillon/) fait moins de
200 Ko : il sert à faire tourner les tests, pas à entraîner un modèle.
Pour un vrai entraînement il faut plusieurs dizaines de gigaoctets.

Cette commande n'automatise rien volontairement : les identifiants exacts des
jeux de données changent, et télécharger 50 Go doit rester une décision
consciente. Voici la marche à suivre.

1. Installer les outils (extra « donnees ») :

       pip install "futo[donnees]"

2. Les sources publiques de français les plus utilisées. Vérifiez l'identifiant
   exact et la licence sur la fiche du jeu de données AVANT de télécharger —
   les noms ci-dessous sont donnés de mémoire et peuvent avoir changé :

   - Wikipédia en français — la source la plus propre, une à deux passes
     recommandées. Licence CC BY-SA : elle impose de citer et de partager à
     l'identique, ce qui a des conséquences sur la licence de vos poids.
     Comptez quelques gigaoctets de texte.
   - FineWeb-2 — extraction filtrée de Common Crawl, couvre le français.
     C'est aujourd'hui le meilleur rapport volume/qualité pour du web.
   - CulturaX, OSCAR, HPLT — autres extractions de Common Crawl, multilingues.
     Qualité variable, filtrage supplémentaire conseillé.
   - Les corpus de l'initiative OpenLLM-France — pensés pour le français,
     avec un travail de sélection déjà fait.

   Ces descriptions valent orientation, pas garantie : contrôlez vous-même le
   volume réel de français, la licence, et la date de collecte.

3. Convertissez au format attendu : un fichier .jsonl (éventuellement .gz), un
   document par ligne, avec un champ « text ».

4. Préparez les shards :

       futo tokenizer entrainer --corpus data/brut/*.jsonl --vocab 32768
       futo data preparer --corpus data/brut/*.jsonl --separateur jsonl

Sur le droit d'auteur et le RGPD : un corpus web contient des textes protégés
et des données personnelles. En France, l'exception de fouille de textes et de
données (article L122-5-3 du code de la propriété intellectuelle) autorise
l'entraînement à des fins de recherche, et pour d'autres usages tant que les
ayants droit ne s'y sont pas opposés. Le règlement européen sur l'IA impose par
ailleurs de publier un résumé suffisamment détaillé des données d'entraînement.
Documentez vos sources au fur et à mesure : reconstituer cette liste après coup
est pénible. Voir docs/DONNEES.md.
"""


# --------------------------------------------------------------------------- #
# train / eval / generer
# --------------------------------------------------------------------------- #


def cmd_train(args) -> int:
    from .config import charger_config
    from .train import entrainer

    cfg = charger_config(args.config, args.set)
    if args.reprendre:
        cfg.train.reprendre = args.reprendre

    resultat = entrainer(cfg, verbeux=not args.silencieux)
    print()
    print(f"Terminé : {resultat.pas_effectues} pas en {resultat.secondes:.1f} s "
          f"({_milliers(resultat.tokens_vus)} tokens).")
    print(f"  perte finale : {resultat.perte_finale:.4f}")
    if math.isfinite(resultat.meilleure_val):
        print(f"  meilleure validation : {resultat.meilleure_val:.4f} "
              f"(perplexité {math.exp(min(resultat.meilleure_val, 20)):.1f})")
    print(f"  checkpoints : {resultat.dossier}")
    return 0


def cmd_eval(args) -> int:
    from .data import ChargeurTokens
    from .eval import mesurer_bits_par_octet, mesurer_perplexite, mesurer_sondes
    from .tokenizer import TokenizerFuto
    from .train import charger, choisir_peripherique

    peripherique = choisir_peripherique()
    modele, charge = charger(args.checkpoint, peripherique)
    cfg = charge["config_objet"]
    print(f"Modèle : {args.checkpoint} (pas {charge['pas']})")
    print(f"  {modele.nombre_parametres() / 1e6:.1f} M paramètres\n")

    tokenizer = TokenizerFuto(args.tokenizer or cfg.data.tokenizer)

    # Perplexité sur la validation, si les shards sont là.
    try:
        chargeur = ChargeurTokens(
            args.donnees or cfg.data.dossier,
            cfg.data.prefixe_val,
            cfg.model.block_size,
            cfg.data.batch_size,
            seed=cfg.data.seed + 1,
        )
        resultat = mesurer_perplexite(modele, chargeur, args.lots, peripherique)
        print(f"Validation ({args.lots} lots) : {resultat}")
    except FileNotFoundError as e:
        print(f"Perplexité non calculée : {e}")

    # Bits par octet sur du texte brut : la mesure comparable entre modèles.
    if args.textes:
        textes = [Path(f).read_text(encoding="utf-8") for f in _etendre(args.textes)]
        resultat = mesurer_bits_par_octet(modele, tokenizer, textes, peripherique=peripherique)
        print(f"Texte brut : {resultat}")

    # Sondes grammaticales françaises.
    chemin_sondes = args.sondes or cfg.eval.sondes
    if Path(chemin_sondes).exists():
        print()
        print(mesurer_sondes(modele, tokenizer, chemin_sondes, peripherique))
    else:
        print(f"\nSondes non trouvées ({chemin_sondes}), étape ignorée.")
    return 0


def cmd_generer(args) -> int:
    import torch

    from .tokenizer import TokenizerFuto
    from .train import charger, choisir_peripherique

    peripherique = choisir_peripherique()
    modele, charge = charger(args.checkpoint, peripherique, restaurer_alea=False)
    cfg = charge["config_objet"]
    tokenizer = TokenizerFuto(args.tokenizer or cfg.data.tokenizer)

    generateur = None
    if args.seed is not None:
        generateur = torch.Generator(device=peripherique).manual_seed(args.seed)

    ids = tokenizer.encoder(args.amorce) if args.amorce else [tokenizer.id_fin]
    amorce = torch.tensor([ids] * args.echantillons, dtype=torch.long, device=peripherique)

    sortie = modele.generer(
        amorce,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        token_fin=tokenizer.id_fin if args.arreter_a_la_fin else None,
        generateur=generateur,
    )
    for i, ligne in enumerate(sortie.tolist()):
        if args.echantillons > 1:
            print(f"\n--- échantillon {i + 1} ---")
        print(tokenizer.decoder(ligne))
    return 0


# --------------------------------------------------------------------------- #
# Assemblage
# --------------------------------------------------------------------------- #


def construire_analyseur() -> argparse.ArgumentParser:
    analyseur = argparse.ArgumentParser(
        prog="futo",
        description="Futo — un modèle de langue français entraîné de zéro.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Pour démarrer, dans l'ordre :

  futo tokenizer entrainer --corpus 'data/echantillon/*.txt' --vocab 4096
  futo data preparer --corpus 'data/echantillon/*.txt'
  futo train configs/futo-tiny.yaml
  futo generer sorties/tiny/dernier.pt --amorce "Il était une fois"

Avant d'engager des jours de calcul, mesurez ce que vaut votre machine :

  futo bench configs/futo-mac.yaml

Chaque commande accepte --set pour surcharger la configuration :

  futo train configs/futo-small.yaml --set train.lr=3e-4 --set model.n_layer=8
""",
    )
    analyseur.add_argument("--version", action="store_true", help="affiche la version")
    sous = analyseur.add_subparsers(dest="commande")

    def ajouter_set(p):
        p.add_argument(
            "--set", action="append", default=[], metavar="CHEMIN=VALEUR",
            help="surcharge une clé de configuration (ex. train.lr=3e-4)",
        )
        return p

    # -- info --
    p = sous.add_parser("info", help="décrit une configuration : taille, mémoire, budget")
    p.add_argument("config", nargs="?", help="fichier YAML de configuration")
    ajouter_set(p)
    p.set_defaults(fonction=cmd_info)

    # -- bench --
    p = sous.add_parser(
        "bench", help="mesure le débit réel de cette machine (sans corpus)"
    )
    p.add_argument("config", help="fichier YAML de configuration")
    p.add_argument("--micro-lots", type=int, default=20,
                   help="nombre de micro-lots chronométrés")
    p.add_argument("--echauffement", type=int, default=3,
                   help="micro-lots ignorés avant la mesure")
    p.add_argument("--silencieux", action="store_true")
    ajouter_set(p)
    p.set_defaults(fonction=cmd_bench)

    # -- tokenizer --
    p_tok = sous.add_parser("tokenizer", help="entraîner et inspecter le tokenizer")
    sous_tok = p_tok.add_subparsers(dest="sous_commande")

    p = sous_tok.add_parser("entrainer", help="entraîne un BPE français")
    p.add_argument("--corpus", nargs="+", default=[CORPUS_EXEMPLE],
                   help="fichiers texte (jokers acceptés)")
    p.add_argument("--sortie", default="data/tokenizer/futo-tokenizer.json")
    p.add_argument("--vocab", type=int, default=32_768, help="taille du vocabulaire")
    p.add_argument("--frequence-min", type=int, default=2)
    p.add_argument("--octets-max", type=int, default=None,
                   help="plafonne le texte lu (inutile d'entraîner un BPE sur plus de 2 Go)")
    p.add_argument("--silencieux", action="store_true")
    p.set_defaults(fonction=cmd_tokenizer_entrainer)

    p = sous_tok.add_parser("info", help="inspecte un tokenizer et mesure sa fertilité")
    p.add_argument("tokenizer", nargs="?", default="data/tokenizer/futo-tokenizer.json")
    p.add_argument("--texte", help="encode ce texte et affiche les tokens")
    p.add_argument("--corpus", nargs="*", help="mesure la fertilité sur ces fichiers")
    p.set_defaults(fonction=cmd_tokenizer_info)

    # -- data --
    p_data = sous.add_parser("data", help="préparer et inspecter les données")
    sous_data = p_data.add_subparsers(dest="sous_commande")

    p = sous_data.add_parser("preparer", help="encode un corpus en shards binaires")
    p.add_argument("--corpus", nargs="+", default=[CORPUS_EXEMPLE])
    p.add_argument("--tokenizer", default="data/tokenizer/futo-tokenizer.json")
    p.add_argument("--sortie", default="data/prepare")
    p.add_argument("--tokens-par-shard", type=int, default=100_000_000)
    p.add_argument("--fraction-val", type=float, default=0.005)
    p.add_argument("--separateur", default="ligne-vide",
                   choices=["ligne-vide", "fichier", "jsonl"])
    p.add_argument("--silencieux", action="store_true")
    p.set_defaults(fonction=cmd_data_preparer)

    p = sous_data.add_parser("info", help="liste les shards d'un dossier")
    p.add_argument("dossier", nargs="?", default="data/prepare")
    p.set_defaults(fonction=cmd_data_info)

    p = sous_data.add_parser("telecharger", help="marche à suivre pour un vrai corpus français")
    p.set_defaults(fonction=cmd_data_telecharger)

    # -- train --
    p = sous.add_parser("train", help="entraîne un modèle")
    p.add_argument("config", help="fichier YAML de configuration")
    p.add_argument("--reprendre", nargs="?", const="auto",
                   help="reprend depuis un checkpoint ('auto' pour le dernier)")
    p.add_argument("--silencieux", action="store_true")
    ajouter_set(p)
    p.set_defaults(fonction=cmd_train)

    # -- eval --
    p = sous.add_parser("eval", help="évalue un checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--tokenizer", help="par défaut : celui de la configuration du checkpoint")
    p.add_argument("--donnees", help="dossier des shards de validation")
    p.add_argument("--textes", nargs="*", help="fichiers texte pour le calcul des bits/octet")
    p.add_argument("--sondes", help="jeu de paires minimales (JSONL)")
    p.add_argument("--lots", type=int, default=50)
    p.set_defaults(fonction=cmd_eval)

    # -- generer --
    p = sous.add_parser("generer", help="produit du texte avec un checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--amorce", default="", help="texte de départ")
    p.add_argument("--tokenizer")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--echantillons", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--arreter-a-la-fin", action="store_true",
                   help="s'arrête au token de fin de document")
    p.set_defaults(fonction=cmd_generer)

    return analyseur


def main(argv: list[str] | None = None) -> int:
    analyseur = construire_analyseur()
    args = analyseur.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"futo {__version__}")
        return 0

    if not hasattr(args, "fonction"):
        # Une sous-commande a été donnée sans sa sous-sous-commande (« futo data »).
        if getattr(args, "commande", None):
            analyseur.parse_args([args.commande, "--help"])
        analyseur.print_help()
        return 1

    try:
        return args.fonction(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nErreur : {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrompu.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
