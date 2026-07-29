"""Futo — un modèle de langue français entraîné de zéro.

Le nom vient de « futé ». Le projet tient en une idée : maîtriser de bout en bout
la chaîne qui va d'un corpus français à un modèle qui parle français, sans boîte
noire, avec un code qu'on peut lire en une soirée.

Point d'entrée habituel : la ligne de commande.

    futo info configs/futo-small.yaml
    futo tokenizer entrainer --corpus data/echantillon/*.txt
    futo data preparer --corpus data/echantillon/*.txt
    futo train configs/futo-tiny.yaml
    futo generer sorties/run/dernier.pt --amorce "Il était une fois"
"""

__version__ = "0.1.0"

from .config import DataConfig, EvalConfig, FutoConfig, ModelConfig, TrainConfig, charger_config
from .model import Futo

__all__ = [
    "__version__",
    "Futo",
    "FutoConfig",
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "EvalConfig",
    "charger_config",
]
