from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

def sampleTypes(selection, fnames):
    """Generate a random sample of tokens and the list of files required to extract them.

    Parameters
    ----------
    selection : dict
        Types to look for as keys, number of tokens to extract from each of them as values.
    settings : str or list
        Selection of filenames to search: either the path to the file with the list, as a string, or the list itself.

    Returns
    -------
    tuple
        A list of token IDs and the list of files where they can be found. Not separated by type.
    """
    if type(fnames) == str:
        with open(fnames, "r") as f:
            fnames = [s.strip() for s in f.readlines()]
        
    last_length = 0
    minimum = 0
    random.shuffle(fnames)
    
    swept_files = set()
    tokens = set()
    final_files = set()
    
    for file in tqdm(fnames):
        if sum(selection.values()) > 0:
            with Path(file).open() as f:
                try:
                    txt = f.readlines()
                except:
                    continue
            """TODO make it flexible adapting to settings like the original code does?"""
            txt = [f"{str(i+1)}\t{x}".split("\t") for i, x in enumerate(txt) if len(x.split("\t")) >= 4]
            findings = { target_type : [x[0] for x in txt if f"{x[3]}/{x[4]}" == target_type]
                        for target_type in selection.keys()
                       if selection[target_type] > 0}
            if len(findings) > 0:
                for target_type in findings.keys():
                    if len(findings[target_type]) > 0:
                        token = f"{target_type}/{Path(file).stem}/{np.random.choice(findings[target_type])}"
                        if not token in tokens:
                            tokens.add(token)
                            final_files.add(file)
                            selection[target_type] -= 1
                            
        else:
            break
            
    return (list(tokens), list(final_files))
