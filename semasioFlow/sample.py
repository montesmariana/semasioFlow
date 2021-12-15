from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import re

from nephosem import CorpusFormatter

def sampleTypes(selection, fnames, settings, oneperfile = True):
    """Generate a random sample of tokens and the list of files required to extract them.

    Parameters
    ----------
    selection : dict
        Types to look for as keys, number of tokens to extract from each of them as values.
    filenames : str or list
        Selection of filenames to search: either the path to the file with the list, as a string, or the list itself.
    settings : dict
        Configuration settings as designed from the `nephosem` workflow.
    oneperfile : bool
        Whether only one token of each lemma can be extracted from the same file.

    Returns
    -------
    tuple
        A list of token IDs and the list of files where they can be found. Not separated by type.
    """
    formatter = CorpusFormatter(settings)
    if type(fnames) == str:
        with open(fnames, "r") as f:
            fnames = [s.strip() for s in f.readlines()]
        
    last_length = 0
    minimum = 0
    random.shuffle(fnames)
    
    tokens = set()
    final_files = set()
    
    for file in tqdm(fnames):
        if sum(selection.values()) <= 0:
            break
        with Path(file).open() as f:
            try:
                txt = [x.strip() for x in f.readlines()]
            except:
                continue
        txt = [(str(i+1), x) for i, x in enumerate(txt) if re.match(settings['line-machine'], x)]
        findings = {
            target_type : _find_type(txt, target_type, formatter)
            for target_type in selection.keys() if selection[target_type] > 0
        }
        if len(findings) == 0:
            continue
        for target_type in findings.keys():
            if len(findings[target_type]) == 0:
                continue
            found_tokens = [np.random.choice(findings[target_type])] if oneperfile else findings[target_type]
            for token in found_tokens:
                tid = f"{target_type}/{Path(file).stem}/{token}"
                if not tid in tokens:
                    selection[target_type] -= 1
                    tokens.add(tid)
                    final_files.add(file)
            
    return (list(tokens), list(final_files))

def _find_type(txt, target_type, formatter):
    return [x[0] for x in txt if formatter.get_type(formatter.match_line(x[1])) == target_type]