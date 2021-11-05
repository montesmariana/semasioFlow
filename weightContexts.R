library(tidyverse)

# Functions ----

lemmaFromTid <- function(tid) paste(str_split(tid, '/')[[1]][-c(3, 4)], collapse = '/')

cleanWord <- function(txt) { # cleaning needed for QLVLNewsCorpus
  str_replace(txt, "and(.+);", "&\\1;") %>%
    str_replace("`", "'") %>% str_replace('</sentence>', '<br>')
}

filterFoc <- function(focparams, tid_data, cw_selection, add_pmi = FALSE){
  # add_pmi: do you want PPMI values as superindices?
  if (str_starts(focparams[[1]], 'LEMMA')) {
    # this is not super reliable, but reliable enough
    max_steps <- if (focparams[[1]] == 'LEMMAPATH2') 2 else 3
    tid_data <- tid_data %>% mutate(
      focsel = same_sentence & steps <= max_steps & cw %in% cw_selection
    )
  } else {
    windows <- parse_integer(str_split(str_extract(focparams[[1]], '\\d+-\\d+'), '-')[[1]])
    # modify pos_sel if you have other parameters :)
    pos_sel <- if (str_ends(focparams[[1]], 'lex')) c('noun', 'adj', 'adv', 'verb') else NA
    tid_data <- tid_data %>% mutate(
      by_bound = str_starts(focparams[[1]], 'nobound') | same_sentence,
      by_win = !is.na(conc_distance) & if_else(side == 'L', conc_distance <= windows[[1]], conc_distance <= windows[[2]]),
      by_pos = str_ends(focparams[[1]], 'all') | pos %in% pos_sel,
      focsel = by_bound & by_win & by_pos
    )
  }
  tid_data <- tid_data %>% 
    mutate(weighted = str_ends(focparams[[2]], 'no') | (!is.na(pmi4) & pmi4 > 0),
           bolden = focsel & weighted,
           sup = if (add_pmi) sprintf('<sup>%s</sup>', round(pmi4, 3)) else '',
           word = if_else(bolden, sprintf('<strong>%s</strong>%s', word, sup), word)
    )
  return(tid_data)
}
getContext <- function(tid, cws, model = 'raw', cw_selection = NA, add_pmi = FALSE) {
  tid_data <- cws %>% filter(token_id == tid) %>% 
    mutate(word = map_chr(word, cleanWord)) %>% 
    filter(word != '<sentence>')
  target <- tid_data %>% filter(side == 'target')  %>% pull(word)
  if (model != 'raw' & !is.na(cw_selection)) {
    if (length(cw_selection) == 1) {
      cw_selection <- str_split(cw_selection, ';')[[1]]
    }
    focparams <- str_split(model, '\\.')[[1]]
    tid_data <- filterFoc(focparams, tid_data, cw_selection, add_pmi)
  }
  
  left <- tid_data %>% filter(side == 'L') %>% arrange(desc(distance)) %>% pull(word) %>% paste(collapse = ' ')
  right <- tid_data %>% filter(side == 'R') %>% arrange(distance) %>% pull(word) %>% paste(collapse = ' ')  
  str_glue('{left} <span class="target">{target}</span> {right}')
  
}

weightContexts <- function(lemma, github_dir,
                           variables_path = file.path(github_dir, paste0(lemma, '.variables.tsv')),
                           cws_detail_path = file.path(github_dir, paste0(lemma, '.cws.detail.tsv')),
                           ppmi_path = file.path(github_dir, paste0(lemma, '.ppmi.tsv')),
                           output_path = variables_path) {
  variables <- read_tsv(variables_path, show_col_types = F, lazy = FALSE)
  models <- variables %>% select(starts_with('_cws.')) %>% 
    rename_all(str_remove, paste0('_cws.', lemma, '.')) %>% 
    colnames()
  
  cws <- read_tsv(cws_detail_path, show_col_types = F)
  ppmi <- read_tsv(ppmi_path, show_col_types = F)
  
  ppmi <- ppmi %>% select(cw, starts_with('pmi_4')) %>% 
    rename_all(str_remove, 'pmi_4_') %>% 
    pivot_longer(-cw, names_to = 'target_lemma', values_to = 'pmi4') %>% 
    filter(!is.na(pmi4))
  
  # account for sentence delimiters
  conc_distance <- cws %>% group_by(token_id, side) %>% arrange(distance) %>%
    filter(!str_starts(word, '<')) %>%
    mutate(conc_distance = seq(n())) %>% ungroup() %>% select(cw_id, conc_distance)
  
  # join everything together
  cws <- cws %>%
    mutate(cw = paste(lemma, pos, sep = '/'),
           target_lemma = map_chr(token_id, lemmaFromTid)) %>% 
    left_join(ppmi, by = c('cw', 'target_lemma')) %>% 
    left_join(conc_distance, by = 'cw_id')
  
  # add raw context
  variables <- variables %>% 
    mutate(`_ctxt.raw` = map_chr(`_id`, getContext, cws = cws))
  
  # weight per model!
  for (m in models) {
    if (!paste('_ctxt', lemma, m, sep = '.') %in% colnames(variables)) {
      variables <- variables %>% 
        mutate(!!paste('_ctxt', lemma, m, sep='.') := map2_chr(
          `_id`, !!sym(paste('_cws', lemma, m, sep='.')),
          getContext, cws = cws, model = m,
          add_pmi = str_ends(m, 'weight'))) # add superindices for PPMIweight only
    }
    
  }
  
  write_tsv(variables, variables_path)
  
}
