import itertools
from numbers import Number
from random import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.dtypes.common import is_string_dtype


class TreeNode:
    # Initialisation de la classe, avec quelques paramètres par défaut
    def __init__(self, min_samples_split=2, max_depth=None, seed=2,
                 verbose=False):
        self.children = {}
        self.decision = None
        self.split_feat_name = None  # Fonction de fractionnement
        self.threshold = None
        # Nombre minimum d'échantillons pour faire un fractionnement
        self.min_samples_split = min_samples_split

        self.max_depth = max_depth
        self.seed = seed
        self.verbose = verbose


# Déclaration de fonctions
# ########################################################
# Cette fonction valide le type de données et la longueur égale des données et de la cible

def validate_data(data, target):
    # Validation du type de données pour l'échantillon (X)
    if not isinstance(data, (list, pd.core.series.Series, np.ndarray, pd.DataFrame)):
        return False

    # Validation du type de données pour l'échantillon (y)
    if not isinstance(target, (list, pd.core.series.Series, np.ndarray)):
        return False
    if len(data) != len(target):
        return False
    return True