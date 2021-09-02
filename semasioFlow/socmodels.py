import os
import pandas as pd
from functools import reduce
import logging

from qlvl import Vocab, TypeTokenMatrix
from qlvl import compute_association, compute_distance
from qlvl.specutils.mxcalc import compute_token_weights, compute_token_vectors

from semasioFlow.utils import fullMerge
from semasioFlow.utils import listCws, countCws

def targetPPMI(targets, vocabs, collocs, type_name = None, main_matrix = None, fname = None, output_dir = None):
    """Registers PPMI values of a target lemma(s) with all possible context words.
    
    Computes PPMI values between target lemma(s) and context words, for weighting.
    It also stores PMI values, raw frequencies and raw co-occurrences.
    
    Parameters
    ----------
    target : list of str
        Lemma(s) for the target(s)
    vocabs : dict
        Vocabularies to extract raw frequency information from; the keys are their names
        and the values are :class:`~qlvl.Vocab`.
    collocs : dict
        Frequency matrices to extract raw co-occurrence frequency and PPMI information from;
        the keys are their names
        and the values are :class:`~qlvl.TypeTokenMatrix`.
    type_name : str, optional
        Name of the type, prefix for file names.
    main_matrix : str, optional
        Key in `collocs` indicating the matrix used for the weighting of the matrix to return.
    fname : str, optional
        Filename to store the frequency data in. By default it combines `type_name` with "ppmi.tsv".
    output_dir = str, optional
        Directory where the matrices will be stored. It is necessary if a `fname` is not provided`.
       
    Returns
    -------
    ppmi : :class:`~qlvl.TypeTokenMatrix`
        Type-level co-occurrence matrix with target type(s) as row(s) and PPMI values
        based the values in `collocs`[`main_matrix`].
    """
    type_name = type_name if type_name else targets[0].split("/")[0]
    main_matrix = main_matrix if main_matrix and main_matrix in collocs.keys() else list(collocs.keys())[0]
    dfs = []
    
    if not output_dir or fname:
        raise ValueError("Please provide `output_dir` or `fname`.")
    if output_dir and not os.path.exists(output_dir):
        logging.info("Creating directory: %s", output_dir)
        os.makedirs(output_dir)
    if not fname:
        fname = f"{output_dir}/{type_name}.ppmi.tsv"
    
    for colloc_name, colloc in collocs.items():
        nfreq = Vocab(colloc.sum(axis=1))
        cfreq = Vocab(colloc.sum(axis=0))
        subcolloc = colloc.submatrix(row = targets).drop(axis = 1, n_nonzero = 0)
        pmi = compute_association(subcolloc, nfreq=nfreq, cfreq=cfreq, meas = 'pmi')
        if colloc_name == main_matrix:
            ppmi = pmi.copy().multiply(pmi > 0).drop(axis = 1, n_nonzero = 0)
        pmidf = pmi.dataframe.transpose()    
        pmidf.columns = [f"pmi_{colloc_name}"] if len(pmidf.columns) == 1 else [f"pmi_{colloc_name}_{x}" for x in pmidf.columns]
        raw_co = subcolloc.dataframe.transpose()
        raw_co.columns = [f"raw_{colloc_name}"] if len(raw_co.columns) == 1 else [f"raw_{colloc_name}_{x}" for x in raw_co.columns]
        dfs.append(fullMerge(pmidf, raw_co))
    
    cws = reduce(fullMerge, dfs)
    for vocab_name, vocab in vocabs.items():
        subvocab = vocab.subvocab(list(cws.index))
        cws[vocab_name] = [subvocab[x] for x in cws.index]
    
    cws.to_csv(fname, sep = '\t', index_label="cw")
    logging.info("Dataframe stored with %s elements at %s.", len(cws.index), fname)
    
    return ppmi

def weightTokens(token_dir, weighting, registers, output_dir = None,
                input_suffix = ".tcmx.bool.pac", output_suffix = ".tcmx.weight.pac"):
    """Apply (or not) weighting to all current token-level matrices across multiple weighting values.
    
    It does store the matrices too.
    
    Parameters
    ----------
    token_dir : str
        Path to the directory where the boolean matrices are stored.
    weighting : dict
        Keys are the names of the PPMI parameter values; values are the matrices to use for weighting,
        (`~qlvl.TypeTokenMatrix`) or `None`.
    register : :class:`pandas.DataFrame`
        Register of model information, with names of the models in the index.
    output_dir : str, optional
        Directory where the matrices will be stored. Defaults to `token_dir`.
    input_suffix : str, default=".tcmx.bool.pac"
        Suffix of the filenames to load.
    output_suffix : str, default=".tcmx.weight.pac"
        Suffix of the filenames to save.
       
    Returns
    -------
    data : dict of pandas.dataframe
        A "model_register" dataframe with one row per model and the parameter settings as columns
        and a "token_register" dataframe with one row per token and the number and lists of context words as columns.
    """
    model_register = {}
    token_register = {}
    output_dir = output_dir if output_dir else token_dir
    
    for focmodel in registers.index:
        input_name = f"{token_dir}/{focmodel}{input_suffix}"
        tokens = TypeTokenMatrix.load(input_name)
        for param, weightMTX in weighting.items():
            modelname = f"{focmodel}.PPMI{param}"
            model_register[modelname] = dict(registers.loc[focmodel])
            model_register[modelname]["foc_pmi"] = param
            output_name = f"{output_dir}/{modelname}{output_suffix}"
            if not weightMTX:
                tokweights = tokens.deepcopy()
            else:
                intersected = list(set(tokens.col_items).intersection(set(weightMTX.col_items)))
                tokweights = compute_token_weights(
                    tokens.submatrix(col = intersected),
                    weightMTX.submatrix(col = intersected)
                ).drop(axis = 0, n_nonzero = 0)
            tokweights.save(output_name)
            model_register[modelname]['tokens'] = len(tokweights.row_items)
            model_register[modelname]['foc_context_words'] = len(tokweights.col_items)
            token_register["_cws." + modelname] = listCws(tokweights)
            token_register["_count." + modelname] = countCws(tokweights)
    data = {
        "model_register" : pd.DataFrame(model_register).transpose(),
        "token_register" : pd.DataFrame(token_register)
    }    
    return data

def createSoc(token_dir, registers, soc_pos, lengths, socMTX,
              output_dir = None,
              input_suffix = ".tcmx.weight.pac", output_suffix = ".tcmx.soc.pac",
             store_focdists = False):
    """Multiply token-by-feature matrix by its second-order matrix.
    
    It does store the matrices too.
    
    Parameters
    ----------
    token_dir : str
        Path to the directory where the boolean matrices are stored.
    register : :class:`pandas.DataFrame`
        Register of model information, with names of the models in the index.
    soc_pos : dict
        The keys are the names of the "SOC-POS" values, the values are filtered `~qlvl.Vocab` objects.
    length : list
        Integer elements will be used to select the `length` most frequent elements in the `soc_pos` lists,
        while other kinds of elements will trigger using the FOC items as SOC items.
    output_dir : str, optional
        Directory where the matrices will be stored. Defaults to `token_dir`.
    input_suffix : str, default=".tcmx.weight.pac"
        Suffix of the filenames to load.
    output_suffix : str, default=".tcmx.soc.pac"
        Suffix of the filenames to save.
    store_focdists : bool or str, default=False
        Whether to store the context-word distance matrix. If False, it doesn't;
        if True, it stores them in `output_dir`; if it's a string, it is taken to be the directory to store them in.
       
    Returns
    -------
    dict of pandas.dataframe
        A register with one row per model and all the parameter settings as columns.
    """
    model_register = {}
    output_dir = output_dir if output_dir else token_dir
    soc_pos = {k : v.get_item_list(sorting = 'freq', descending = True) for k, v in soc_pos.items()}
    nfreq = Vocab(socMTX.sum(axis=1))
    cfreq = Vocab(socMTX.sum(axis=0))
    
    soc_params = [(sp, length) for sp in soc_pos for length in lengths]
    for focmodel in registers.index:
        input_name = f"{token_dir}/{focmodel}{input_suffix}"
        tokens = TypeTokenMatrix.load(input_name)
        for sp, length in soc_params:
            modelname = f"{focmodel}.LENGTH{length}.SOCPOS{sp}"
            model_register[modelname] = dict(registers.loc[focmodel])
            model_register[modelname]["soc_length"] = length
            model_register[modelname]["soc_pos"] = sp
            output_name = f"{output_dir}/{modelname}{output_suffix}"
        
            sp_list = soc_pos[sp]
            soc_cols =  sp_list[:length] if type(length) == int else list(set(tokens.col_items).intersection(set(sp_list)))
            
            socMTX_sub = socMTX.submatrix(row = tokens.col_items, col = soc_cols)
            soc_pmi = compute_association(socMTX_sub, nfreq=nfreq, cfreq=cfreq, meas = 'ppmi')
            if store_focdists:
                focdists_dir = store_focdists if type(store_focdists) == str else output_dir
                if not os.path.exists(focdists_dir):
                    logging.info("Creating directory: %s", focdists_dir)
                    os.makedirs(focdists_dir)
                focdists_fname = f"{focdists_dir}/{modelname}.wwmx.dist.csv"
                compute_distance(soc_pmi).to_csv(focdists_fname)
            tokvecs = compute_token_vectors(tokens, soc_pmi)
            tokvecs.save(output_name)
    return pd.DataFrame(model_register).transpose()          
        