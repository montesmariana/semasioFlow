import re
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from nephosem import CorpusFormatter
from nephosem.core.graph import SentenceGraph

def sameSentence(here, target, delimiters):
    """Determine whether two tokens are in the same sentence.
    
    Parameters
    ----------
    here : int
        Index of the context word
    target : int
        Index of the target word
    delimiters : list of int
        Indices of the sentence delimiters
        
    Returns
    -------
    bool
        Whether the two words are in the same sentence, i.e. not separated by a delimiter.
    """
    if len(delimiters) == 0:
        return True
    start = min(here, target)
    end = max(here, target)
    return not bool(sum([x in range(start, end) for x in delimiters]))

def listContextwords(type_name, tokenlist, fnames, settings, left_win = None, right_win = None):
    """Create dataframe with detail on context words of tokens.
    
    It includes the elements that `global_columns` and `line_machine` extract from the corpus
    along with the distance (and side) to the target and whether they occur in the same sentence.
    
    Parameters
    ----------
    type_name : str
        Name of the type
    tokenlist : list of str
        List of token IDs
    fnames : list of str
        List of file names to find the tokens in
    settings : dict
        Settings as created for the full workflow
    left_win : int, optional
        Number of context words to extract from the left side, including sentence delimiters. Defaults to the settings values.
    right_win : int, optional
        Number of context words to extract from the right side, including sentence delimiters. Defaults to the settings values.
    
    Returns
    -------
    :class:`pandas.DataFrame`
        Data frame with one row per context word per token, information from the corpus and information relative to the target.
    
    """
    left_win = left_win + 1 if left_win else settings['left-span'] + 1
    right_win = right_win + 1 if right_win else settings['right-span'] + 1
    
    formatter = CorpusFormatter(settings)
    text_variables = formatter.global_columns
    useDep = formatter.edge_attr in text_variables
    cws = {}
    basic_dict = {'target_lemma' : type_name}

    for file in tqdm(fnames):
        tokens = [(int(tokid.split('/')[3])-1, tokid)
                  for tokid in tokenlist if tokid.split("/")[2] == Path(file).stem]
        with open(file, 'r', encoding = 'latin-1') as f:
            lines = [s.strip() for s in f.readlines()]
            
        for index, tokid in tokens:
            tokendict = basic_dict.copy()
            tokendict.update({'token_id' : tokid})
            span = range(max(0, index-left_win), min(index+right_win, len(lines)))
            not_text_lines = [i for i in span if not formatter.match_line(lines[i])]
            if useDep:
                steps = {str(x['this']) : x for x in getSteps(lines, index, formatter)}

                
            for i in span:
                dist = abs(i-index)
                if dist == 0:
                    side = position = "target"
                else:
                    side = "L" if index > i else "R"
                    position = side + str(dist-1)
                cwid = tokid + "/" + position
                cwdict = tokendict.copy()
                cwdict.update({
                    'distance' : dist,
                    'side' : side,
                    'position' : position
                })
                match = formatter.match_line(lines[i])
                if match:
                    text_values = {k:v for k, v in zip(text_variables, match.groups())}
                    text_values['cw'] = formatter.get_type(match)
                    ss = sameSentence(i, index, not_text_lines)
                    text_values['same_sentence'] = ss
                    matchdict = cwdict.copy()
                    matchdict.update(text_values)
                    if ss and useDep:
                        if dist == 0:
                            path_data = {'steps' : 0, 'path' : "#T", 'rep_path' : "#T"}
                        else:
                            this_path = steps[str(getIdx(lines[i], formatter))]
                            path_data = {k : v for k, v in this_path.items() if k in ['steps', 'path', 'rep_path']}
                        matchdict.update(path_data)
                    cws[cwid] = matchdict
                else:
                    text_values = {k:lines[i] for k in text_variables}
                    text_values['same_sentence'] = False
                    matchdict = cwdict.copy()
                    matchdict.update(text_values)
                    cws[cwid] = matchdict
    cws = pd.DataFrame(cws).transpose()
    return cws

def findLabel(target_lid, cw_lid, sent, goal, explicit):
    """Label item in dependency path.
    
    Parameters
    ----------
    target_lid : int
        Index of target token.
    cw_lid : int
        Index of context token to add to the path.
    sent : :class:`~nephosem.SentenceGraph`
        Sentence
    goal : int
        Index of the item we want to replace
    explicit : bool
        Whether the path is explicit or not; implicit paths replace the final context word with "CW".
        
    Returns
    -------
    str
        Replacement string to add to path
    """
    if target_lid == goal:
        return "#T"
    elif not explicit and cw_lid == goal:
        return "Cw"
    else:
        return sent.nodes[goal].get("lemma", "")
    
def drawPath(feature, target, head, dependent, rel, sent,
             steps, explicit = False):
    """Find path between a feature and the target in a sentence.
    
    The longer and more convoluted the path, the less reliable.
    
    Parameters
    ----------
    feature : int
        Index of the context word
    target : int
        Index of the target token
    head : int
        Index of the head of the path
    dependent : int
        Index of the dependent of the path
    rel : :class: `nx.Graph.edge`
        Relationship between the head and the dependent
    sent : :class: `~nephosem.SentenceGraph`
    steps : list of dict
        List of parent and dependent relations
    explicit : bool, default=False
        Whether the lemma of the feature should be made explicit
    
    Returns
    -------
    path : str
        Written path between target and feature
    """
    # set up tag for head
    h = findLabel(target, feature, sent, head, explicit)
    d = findLabel(target, feature, sent, dependent, explicit)
    d = rel + ":" + d
    
    if target == head or target == dependent:
        # for one step, direct relationship
        path = h + "->" + d
    else:
        parent = [x for x in steps if x["this"] == head or x["this"] == dependent][0]
        # extract the other element in steps centered around the actual head or dependent
        # Then we use their explicit path to draw the path?
        if dependent == parent["head"]:
            d = rel + ":" + parent["path"]
            path = h + "->" + d
        elif head == parent["dep"]:
            parent_dep = parent["rel"] + ":" + sent.nodes[head].get("lemma", "")
            path = parent["path"].replace(parent_dep, parent_dep + "->" + d)
        else:
            old_path = parent["path"].rsplit("->", 1) # this is not entirely reliable
            path = old_path[0] + "->[" + old_path[1] + "," + d + "]"
    return path

def joinDeps(feature, head, dependent, sent, step_dist = 0, target = None, steps = None):
    """Summarize dependency path"""
    rel = sent.edges[(head, dependent)]["deprel"]
    kwargs = {"feature" : feature, "head" : head, "dependent" : dependent, "rel" : rel,
             "sent" : sent, "steps" : steps, "target" : target}
    explicit_path = drawPath(explicit = True, **kwargs)
    implicit_path = drawPath(explicit = False, **kwargs)
    return {"this" : feature, "head": head,
           "dep" : dependent, "steps" : step_dist,
           "rel": rel, "path" : explicit_path,
            "rep_path" : implicit_path}

def getIdx(line, formatter):
    """Get id of a line in a sentence.
    
    Parameters
    ----------
    line : str
    formater : :class: `~nephosem.CorpusFormatter`
    
    Returns
    -------
    int
    """
    line_machine = formatter.line_machine
    match = re.match(line_machine, line)
    return int(formatter.get(match, 'id')) if match else 0.5

def getSteps(text, target_idx, formatter):
    """Return paths between each context word and the target.
    
    Parameters
    ----------
    text : list
        Lines of the corpus
    target_idx : int
        Index of target token
    formatter : :class:`~nephosem.CorpusFormatter`
    
    Returns
    -------
    list of dict
    """
    target_line = text[target_idx] # line corresponding to the target
    target_lid = getIdx(target_line, formatter) # index of target within sentence
    
    ss = [(i+(target_idx-target_lid), x) for i, x in enumerate(text)
          if getIdx(x, formatter)-target_lid == i-target_idx] # lines of the sentence of the target
    sent = SentenceGraph(sentence=ss, formatter=formatter)
    subset = [v for v, vitem in sent.nodes if v != target_lid] # nodes matching the target
    steps = []
    while len(subset) > 0:
        kwargs = {"sent":sent, "step_dist" : len(steps)+1, "target" : target_lid}
        if len(steps) == 0:
            predecessors = [joinDeps(feature = x, head = x, dependent = target_lid, **kwargs) for x in subset
                            if x in sent.predecessors(target_lid)]
            successors = [joinDeps(feature = x, head = target_lid, dependent = x, **kwargs) for x in subset
                          if x in sent.successors(target_lid)]
            steps.append(predecessors + successors)
            subset = [x for x in subset if not x in [y["this"] for y in steps[-1]]]
        else:
            kwargs["steps"] = steps[-1]
            predecessors = [joinDeps(feature = x, head = x, dependent = y["this"], **kwargs) for x in subset for y in steps[-1]
                            if x in sent.predecessors(y["this"])]
            successors = [joinDeps(feature = x, head = y["this"], dependent = x, **kwargs) for x in subset for y in steps[-1]
                          if x in sent.successors(y["this"])]
            steps.append(predecessors + successors)
            subset = [x for x in subset if not x in [y["this"] for y in steps[-1]]]
    return [x for y in steps for x in y]