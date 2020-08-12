# -----------------------------------------------------------------------------------------------------------------------------------------------------------
#   Methods that allow the evaluation process of the model.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It contains all the methods needed for the computation of the Spearman score based on the cosine_similarity for the evaluation of the model.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import numpy as np

# --- Function that retrieves all the embeddings of all the senses associated to a certain lemma. ---
# :param word: lemma
# :param luTable: the look-up table of the embeddings, the keys are the senses and the values are the corr. embeddings
# :param word2senses: dictionary whose keys are the lemmas and the corr. values are lists containing all the BabelNet ids corr. to that lemma

def get_senses(word, luTable, word2senses):

    emb_Set = []
    senses = word2senses[word]      # gets the list of all BabelNet ids of the lemma
    for sense in senses:
        lemma_syn = word+'_'+sense  # builds the sense
        if lemma_syn in luTable:    # if there is an embedding of this sense retrieve it
            emb_Set.append(luTable[lemma_syn])
    return np.asarray(emb_Set)      # converts the list as array


# --- Function that computes the maximum cosine similarity among all the embeddings of all the senses of two words. ---
# :param w1: first word
# :param w2: second word
# :param luTable: the look-up table of the embeddings, the keys are the senses and the values are the corr. embeddings
# :param word2senses: dictionary whose keys are the lemmas and the corr. values are lists containing all the BabelNet ids corr. to that lemma

def get_cosine(w1, w2, luTable, word2senses):

    if w1 in word2senses and w2 in word2senses:     # if there are senses for these words
        S1 = get_senses(w1, luTable, word2senses)   # gets the senses for w1
        S2 = get_senses(w2, luTable, word2senses)   # gets the senses for w2
        if S1 != [] and S2 != []:                   # if there are embeddings for the senses then            
            return np.max(cosine_similarity(S1, S2))    # compute the maximum cosine similarity
        else:
            return -1.0                             # otherwise return the worst case
    else:
        return -1.0


# --- Function that computes the score of the model. ---
# :param path: the path of the file for the evaluation (combined.tab)
# :param luTable: the look-up table of the embeddings, the keys are the senses and the values are the corr. embeddings
# :param word2senses: dictionary whose keys are the lemmas and the corr. values are lists containing all the BabelNet ids corr. to that lemma

def score_model(path, luTable, word2senses):

    gold = []               # list of the gold scores
    cosine = []             # list of cosine scores
    f = open(path, encoding='utf-8')
    line = f.readline()     # discard first line (comments)
    line = f.readline()

    while line!='':
        parts = line.split()
        cos = get_cosine(parts[0].lower(), parts[1].lower(), luTable, word2senses)  # get cosine score of the two words
        cosine.append(cos)              # updates cosine list
        gold.append(float(parts[2]))    # updates gold list
        line = f.readline()
    
    f.close()
    rho, _ = spearmanr(gold, cosine)    # computes the Spearman score
    return rho