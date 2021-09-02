from copy import deepcopy
from functools import reduce
import pandas as pd
import os.path
from tqdm import tqdm
import logging

from qlvl import TokenHandler, TypeTokenMatrix # to generate frequency lists and matrices
from qlvl.models.deprel import DepRelHandler
from qlvl.specutils.mxutils import merge_two_matrices

from semasioFlow.utils import booleanize, listCws, countCws

def createBow(query, settings, type_name = None,
              fnames = None, foc_win = None, foc_pos = { "all" : []},
              bound = { "match" : "<artikel>", "values" : [False]},
              tokenlist = None, dummy_sentbound = "<artikel>",
             suffix = ".tcmx.bool.pac",
             output_dir = None):
    """Create multiple bag-of-words token-level models on a loop.
    
    Parameters
    ----------
    query : :class:`~qlvl.Vocab`
        Types to collect tokens from
    settings : dict
    type_name : str, optional
        Name of the type, prefix for file names
    fnames : str or list, optional
        Path to list of filenames or list of filenames to search tokens in.
        Default is the full corpus.
    foc_win : list of tuples, optional
        List of window size settings; each tuple is one setting indicating left and right spans correspondingly.
        The default value is the one in the settings.
    foc_pos : dict, optional
        The keys are the labels of the part-of-speech settings and the values
        are lists of context words to filter the matrix.
        The default value is "all", with no filters.
    bound : dict, optional
        The `match` value indicates the regex for the sentence boundary and
        a list of boolean values indicating whether it is respected.
        The default is with `"<artikel>"` as regex and no consideration for sentence boundaries.
    tokenlist : list, optional
        List of token ID's to filter the matrix. By default, the rows are not filtered.
    dummy_sentbound : str, default="<artikel>"
        String that will not match anything relevant in the corpus
        and therefore cancels sentence boundaries.
    suffix : str, default=".ttmx.pos.pac"
        Suffix for the filenames of the position matrices.
    output_dir = str, optional
        Directory where the matrices will be stored.
        By default it's a subdirectory `type_name` within the subdirectry "tokens"
        within `settings['output-path']`. If the directory does not exist it will be created.   
        
    Returns
    -------
    pandas.DataFrame
        Register of model parameters: it has one row per model and the parameter settings as columns.
        
    Note
    ----
    As a secondary effect, the function stores all the token-by-feature boolean matrices.    
    """
    
    default_settings = deepcopy(settings)
    foc_win = foc_win if foc_win else [(settings['left-span'], settings['right-span'])]
    type_name = type_name if type_name else query.get_item_list()[0].split("/")[0]
    output_dir = output_dir if output_dir else f"{settings['output-path']}/tokens/{type_name}/"
    
    if not os.path.exists(output_dir):
        logging.info("Creating directory: %s", output_dir)
        os.makedirs(output_dir)
        
    model_register = {}
    
    window_boundaries = [(w, b) for w in foc_win for b in bound["values"]]
    for w, b in tqdm(window_boundaries):
        settings['left-span'] = w[0]
        settings['right-span'] = w[1]
        settings['separator-line-machine'] = bound["match"] if b else dummy_sentbound
        tokhan = TokenHandler(query, settings=settings)
        tokens = tokhan.retrieve_tokens(fnames = fnames)
        for fp, pos_list in foc_pos.items():
            cols = pos_list if len(pos_list) > 0 else tokens.col_items
            rows = tokenlist if tokenlist else tokens.row_items
            toks = booleanize(tokens.submatrix(row = rows, col = cols)).drop(axis = 0, n_nonzero = 0)
            modelname = f"{type_name}.{'no' if not b else ''}bound{w[0]}-{w[1]}{fp}"
            model_register[modelname] = {
                "foc_base" : "BOW",
                "foc_win" : f"{w[0]}-{w[1]}",
                "foc_pos" : fp,
                "bound" : b
            }
            filename = f"{output_dir}/{modelname}{suffix}"
            toks.save(filename)
            
    settings = default_settings
    return pd.DataFrame(model_register).transpose()

def tokensFromMacro(query, macros, settings, fnames = None, weight = 1):
    """Obtain dependency-based token-level model.
    
    Parameters
    ----------
    query : :class:`~qlvl.Vocab`
        Types to collect tokens from.
    macros : list of :class:~qlvl.core.graph.MacroGraph
        Can be obtained with SemasioFlow.load.loadMacro().
    settings : dict
        It MUST include an appropiate 'separator-line-machine' value.
    fnames : str or list, optional
        Path to list of filenames or list of filenames to search tokens in. Default is the full corpus.
    weight : int, default=1
        Constant to multiply the values for (for weighting mechanisms).
        
    Returns
    -------
    res : :class:`~qlvl.TypeTokenMatrix`
        Token level boolean matrix.
    """
    dephan = DepRelHandler(settings, workers=4, targets=query, mode='token')
    dephan.read_templates(macros=macros)

    tokens = dephan.build_dependency(fnames=fnames)
    return TypeTokenMatrix(tokens.matrix*weight, tokens.row_items, tokens.col_items)

def createRel(query, settings, rel_macros, type_name = None,
              fnames = None, tokenlist = None, foc_filter = None,
             suffix = ".tcmx.bool.pac", output_dir = None):
    """Create multiple LEMMAREL token-level models on a loop.
    
    Parameters
    ----------
    query : :class:`~qlvl.Vocab`
        Types to collect tokens from
    settings : dict
    rel_macros : list of tuples
        Each tuple is a LEMMAREL group.
        The first element of each tuple is its label (for the name of the model).
        The second element is a list of :class:~qlvl.core.graph.MacroGraph,
        which can be obtained with SemasioFlow.load.loadMacro().
    type_name : str, optional
        Name of the type, prefix for file names
    fnames : str or list, optional
        Path to list of filenames or list of filenames to search tokens in.
        Default is the full corpus.
    tokenlist : list, optional
        List of token ID's to filter the matrix. By default, the rows are not filtered.
    foc_filter : list, optional
        List of context words to filter the matrix. By default, the columns are not filtered.
    suffix : str, default=".ttmx.pos.pac"
        Suffix for the filenames of the position matrices.
    output_dir = str, optional
        Directory where the matrices will be stored.
        By default it's a subdirectory `type_name` within the subdirectry "tokens"
        within `settings['output-path']`. If the directory does not exist it will be created.   
        
    Returns
    -------
    pandas.DataFrame
        Register of model parameters: it has one row per model and the parameter settings as columns.
        
    Note
    ----
    As a secondary effect, the function stores all the token-by-feature boolean matrices.    
    """
    type_name = type_name if type_name else query.get_item_list()[0].split("/")[0]
    output_dir = output_dir if output_dir else f"{settings['output-path']}/tokens/{type_name}/"
    
    if not os.path.exists(output_dir):
        logging.info("Creating directory: %s", output_dir)
        os.makedirs(output_dir)
    
    model_register = {}
    
    for rel_name, macros in rel_macros:
        tokens = tokensFromMacro(query, macros, settings, fnames)
        rows = tokenlist if tokenlist else tokens.row_items
        cols = foc_filter if foc_filter else tokens.col_items
        
        toks = booleanize(tokens.submatrix(row = rows, col = cols)).drop(axis = 0, n_nonzero = 0)
    
        modelname = f"{type_name}.{rel_name}"
        model_register[modelname] = {
            "foc_base" : "LEMMAREL",
            "LEMMAREL" : rel_name
        }
        filename = f"{output_dir}/{modelname}{suffix}"
        toks.save(filename)
        
    return pd.DataFrame(model_register).transpose()

def createPath(query, settings, path_macros, type_name = None,
              fnames = None, tokenlist = None, foc_filter = None,
             suffix = ".tcmx.bool.pac", output_dir = None):
    """Create multiple PATH token-level models on a loop.
    
    Parameters
    ----------
    query : :class:`~qlvl.Vocab`
        Types to collect tokens from
    settings : dict
    path_macros : list of tuples
        Each tuple is a LEMMAPATH group.
        The first element of each tuple is its label (for the name of the model).
        The second element is a list of :class:~qlvl.core.graph.MacroGraph,
        which can be obtained with SemasioFlow.load.loadMacro().
        The third element is a boolean indicating whether the weight of each template.
    type_name : str, optional
        Name of the type, prefix for file names
    fnames : str or list, optional
        Path to list of filenames or list of filenames to search tokens in.
        Default is the full corpus.
    tokenlist : list, optional
        List of token ID's to filter the matrix. By default, the rows are not filtered.
    foc_filter : list, optional
        List of context words to filter the matrix. By default, the columns are not filtered.
    suffix : str, default=".ttmx.pos.pac"
        Suffix for the filenames of the position matrices.
    output_dir = str, optional
        Directory where the matrices will be stored.
        By default it's a subdirectory `type_name` within the subdirectry "tokens"
        within `settings['output-path']`. If the directory does not exist it will be created.   
        
    Returns
    -------
    pandas.DataFrame
        Register of model parameters: it has one row per model and the parameter settings as columns.
        
    Note
    ----
    As a secondary effect, the function stores all the token-by-feature boolean matrices.    
    """
    type_name = type_name if type_name else query.get_item_list()[0].split("/")[0]
    output_dir = output_dir if output_dir else f"{settings['output-path']}/tokens/{type_name}/"
    
    if not os.path.exists(output_dir):
        logging.info("Creating directory: %s", output_dir)
        os.makedirs(output_dir)
    
    model_register = {}
    
    def mergeTwo(mat1, mat2):
        shared_tokens = list(set(mat1.row_items).intersection(set(mat2.row_items)))
        return merge_two_matrices(mat1.submatrix(row = shared_tokens), mat2.submatrix(row = shared_tokens))
    
    for path_name, macros, weights in path_macros:
        weights = weights if weights else [1 for _ in range(len(macros))]
        token_matrices = [
            tokensFromMacro(query, macro, settings, fnames, weight)
            for macro, weight in zip(macros, weights)
        ]
        tokens = reduce(mergeTwo, token_matrices)
        rows = tokenlist if tokenlist else tokens.row_items
        cols = foc_filter if foc_filter else tokens.col_items
        
        toks = booleanize(tokens.submatrix(row = rows, col = cols)).drop(axis = 0, n_nonzero = 0)
    
        modelname = f"{type_name}.{path_name}"
        model_register[modelname] = {
            "foc_base" : "LEMMAPATH",
            "LEMMAPATH" : path_name
        }
        filename = f"{output_dir}/{modelname}{suffix}"
        toks.save(filename)
        
    return pd.DataFrame(model_register).transpose()
