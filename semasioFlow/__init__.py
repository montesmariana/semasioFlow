import pandas as pd # for concordances
import numpy as np # for "booleanize()"
from scipy import sparse # for "booleanize()"
import os
import sys
# import re
# import json
# from tqdm import tqdm
# from pathlib import Path

#qlvldir = "/home/projects/semmetrix/mariana_wolken/depmodel"
qlvldir = "../../../enzocxt/depmodel"
sys.path.append(qlvldir)

from qlvl.conf import ConfigLoader # to setup the configuration
# from qlvl import Vocab, TypeTokenMatrix # to manage frequency lists and matrices
# from qlvl import ItemFreqHandler, ColFreqHandler, TokenHandler # to generate frequency lists and matrices
# from qlvl import compute_association, compute_distance # to compute PPMI and distances
# # compute_cosine would return similarities (I think), compute_simrank, similarity ranks
# from qlvl.specutils.mxcalc import compute_token_weights, compute_token_vectors # for token level. I have incorporated Stefano's modifications in my copy, with adjustments
# # from qlvl.basics.mxcalc import compute_token_weights, compute_token_vectors # non depmodel version
# from qlvl.models.typetoken import build_tc_weight_matrix # for weighting at token level

# # For dependencies
# from qlvl.core.graph import SentenceGraph, MacroGraph, PatternGraph
# from qlvl.models.deprel import DepRelHandler, read_sentence
