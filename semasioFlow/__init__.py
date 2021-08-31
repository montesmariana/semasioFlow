import pandas as pd # for concordances
import numpy as np # for "booleanize()"
from scipy import sparse # for "booleanize()"
import os
import sys

qlvldir = "/home/projects/semmetrix/mariana_wolken/depmodel"
#qlvldir = "../../../enzocxt/depmodel"
sys.path.append(qlvldir)

from qlvl.conf import ConfigLoader # to setup the configuration
