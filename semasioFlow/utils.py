import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

from qlvl import TypeTokenMatrix
from qlvl.specutils.deputils import parse_pattern, draw_tree, draw_match, draw_labels, get_root


def booleanize(mtx, include_negative = True):
    """Transform matrix into matrix of 1's and 0's.

    Parameters
    ----------
    mtx : :class:`~qlvl.TypeTokenMatrix`
    include_negative : bool
        Whether negative values should be transformed to 1's.

    Returns
    -------
    :class:`~qlvl.TypeTokenMatrix`
    """
    # For PPMI matrices, include_negative should be False
    boolean_array = mtx.matrix.toarray() != 0 if include_negative else mtx.matrix.toarray() > 0
    boolean_sparse = sparse.csr_matrix(boolean_array.astype(np.int))
    return TypeTokenMatrix(boolean_sparse, mtx.row_items, mtx.col_items).drop(axis = 1, n_nonzero = 0)

def listCws(tokens):
    """List the context words co-occurring with each token in a matrix.

    Parameters
    ----------
    tokens : :class:`~qlvl.TypeTokenMatrix`
        (Boolean) token-level matrix to count context words from
    
    Returns
    -------
    dict
        Keys are token IDs and values are `;`-separated lists of context words with nonzero values in that matrix.
    """    
    return {r : ";".join([cw for cw in tokens.col_items if tokens[r, cw] != 0]) for r in tokens.row_items}

def countCws(tokens):
    """Count the context words co-occurring with each token in a matrix.

    Parameters
    ----------
    tokens : :class:`~qlvl.TypeTokenMatrix`
        (Boolean) token-level matrix to count context words from
    
    Returns
    -------
    dict
        Keys are token IDs and values are the number of context words with nonzero values in that matrix.
    """    
    return {r : len([cw for cw in tokens.col_items if tokens[r, cw] != 0]) for r in tokens.row_items}

def plotPatterns(macros):
    """Visualize dependency macros as graphs.

    Parameters
    ----------
    macros : list of :class:~qlvl.core.graph.MacroGraph
        Can be obtained with SemasioFlow.load.loadMacro().
    
    """    
    plt.rcParams['figure.figsize'] = (20.0, 32.0)
    for i in range(len(macros)):
        plt.subplot(5, 2, i+1)
        draw_labels(macros[i].graph, v_labels='pos', e_labels='deprel')
    plt.show()

def fullMerge(df1, df2):
    """Wrapper for outer merge of pandas DataFrames by index.

    Parameters
    ----------
    df1 : :class:`pandas.DataFrame`
    df2 : :class:`pandas.DataFrame`
    
    Returns
    -------
    :class:`pandas.DataFrame`
        Outer merge by index of both dataframes.
    """   
    return pd.merge(df1, df2, how = "outer", left_index = True, right_index = True)