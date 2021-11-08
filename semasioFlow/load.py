import os
import pandas as pd
from functools import reduce
import logging

from nephosem import Vocab, TypeTokenMatrix
from nephosem import ItemFreqHandler, ColFreqHandler
from nephosem.core.graph import MacroGraph, PatternGraph

def loadVocab(fname, settings, fnames = None):
    """Load an existing vocabulary or create one.

    Parameters
    ----------
    fname : str
        Path where an existing vocabulary is stored or where it would be stored.
    fnames : str or list, optional
        Corpus file names
    settings : dict
        Settings for creating the vocabulary and to extract the encoding information.

    Returns
    -------
    vocab : :class:`~nephosem.Vocab`

    Note
    ----
    If the file does not exist, it creates it and stores it in the filename given.
    """
    if os.path.exists(fname):
        logging.info("Loading existing vocabulary...")
        return Vocab.load(fname, encoding = settings['outfile-encoding'])
    else:
        logging.info("Creating new vocabulary...")
        ifhan = ItemFreqHandler(settings = settings)
        vocab = ifhan.build_item_freq(fnames = fnames)
        vocab.save(fname, encoding = settings['outfile-encoding'])
        return vocab
    
def loadMacro(templates_dir, graphml_name, macro_name):
    """Load patterns and templates to create dependency-based models.
    
    The output can be used as `macro` argument for nephosem.depmodel.DepHandler objects.

    Parameters
    ----------
    templates_dir : str
        Directory where the templates are stored.
    graphml_name : str
        Basename of the pattern file (before the ".template.graphml" extension).
    macro_name : str
        Basename of the feature-template file (before the ".target-feature-macro.xml" extension).

    Returns
    -------
    list of :class:`~nephosem.core.graph.MacroGraph`
    """
    graphml_fname = f"{templates_dir}/{graphml_name}.template.graphml"
    patterns = PatternGraph.read_graphml(graphml_fname)
    macro_fname = f"{templates_dir}/{macro_name}.target-feature-macro.xml"
    return MacroGraph.read_xml(macro_fname, patterns)

def loadColloc(fname, settings, row_vocab = None, fnames = None, col_vocab = None):
    """Load an existing vocabulary or create one.

    Parameters
    ----------
    fname : str
        Path where an existing vocabulary is stored or where it would be stored.
    settings : dict
        Settings for creating the vocabulary and to extract the encoding information.
    fnames : str or list, optional
        Corpus file names
    row_vocab : :class:`~nephosem.Vocab`, optional if fname exists
        Vocabulary for the rows of the collocation matrix.
    col_vocab : :class:`~nephosem.Vocab`, optional
        Vocabulary for the columns of the collocation matrix.

    Returns
    -------
    freqMTX : :class:`~nephosem.TypeTokenMatrix`
        Type-level co-occurrence matrix matrix.

    Note
    ----
    If the file does not exist, it creates it and stores it in the filename given.
    """
    if os.path.exists(fname):
        logging.info("Loading existing collocation matrix...")
        return TypeTokenMatrix.load(fname)
    else:
        if row_vocab is None:
            logging.error("You need to specify a row vocabulary to create a new matrix")
            return
        logging.info("Creating new collocation matrix...")
        cfhan = ColFreqHandler(settings = settings, row_vocab = row_vocab, col_vocab = col_vocab)
        freqMTX = cfhan.build_col_freq(fnames = fnames)
        freqMTX.save(fname)
        return freqMTX

def loadFocRegisters(register_path, type_name, prefixes = ["bow", "rel", "path"]):
    """Load and combine first-order register dataframes.

    Parameters
    ----------
    register_path : str
        Directory where the dataframes are stored.
    type_name : str
        First part of the file names.
    prefixes : list of str
        Infixes in the filenames
        
    Returns
    -------
    registers : :class:`pandas.DataFrame`
        Merged register dataframes
    """
    
    def loadReg(prefix):
        with open(f"{register_path}/{type_name}.{prefix}-models.tsv", "r") as f:
            reg = pd.read_csv(f, sep = "\t")
        return reg
    
    def myMerge(df1, df2):
        return df1.merge(df2, how = "outer")
    
    registers = [loadReg(x) for x in prefixes]
    registers = reduce(myMerge, registers)
    registers = registers.set_index("_model")
    return registers
