# -----------------------------------------------------------------------------------------------------------------------------------------------------------
#  Methods that fix all the solvable inconsistences as to provide a stuctured-shaped input tensor.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It creates the stuctured-input tensors, as the one in input_utils.py but fixing all the inconsistencies of the dataset so as to augment the data provided. 
# It also handles a higher number of constraints over the validation process of words.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import json

punctuation = ['.', ',', ':', ';', '"', "'", '!', '$', '£', '%', '&', '/', '(', ')', '=', '?', '^', '-', '_', '|', '<', '>', '+', '-', '*']

with open("./stopwords.json") as handle:        # imports stopword
    stopwords = set(json.load(handle))

with open("./long_stopwords.json") as handle:   # imports longer stopwords
    long_stopwords = stopwords.union(set(json.load(handle)))

with open("./1words.json") as handle:           # imports the valid 1-char. words
    word1 = json.load(handle)

with open("./2words.json") as handle:           # imports the valid 2-chars words
    word2 = json.load(handle)


# --- Function that verifies whether a string is valid or not. ---
# :param elem: the string to verify
# :return boolean: True if the string is not a punct. symbol, nor a digit, nor null, nor a stopword and nor a not valid 1 or 2 chars word, False otherwise

def isValid(elem):

    if elem in punctuation:
        return False
    if elem.isdigit():
        return False
    
    if not elem:
        return False
    if elem in long_stopwords:
        return False
    if len(elem)==1 and elem not in word1:
        return False
    if len(elem)==2 and elem not in word2:
        return False
    
    return True


# --- Function that builds a row for the input tensor. ---
# :param sentence: the sentence
# :param a: (anchor, lemma, id_synset) of an annotation of sentence
# :param windowSize: int representing the window-size
# :return row: a list of length 2*windowSize +1 centered in the sense surrounded with 2*windowSize valid lower-case elements, eventually padded

def fix_row(sentence, a, windowSize):

    anchor = a[0].lower()               # gets lower-case anchor (first source of inconsistence (S.O.I.) solved)
    lemma = a[1].lower()                # gets lower-case lemma
    lemma = lemma.replace(' ', '_')     # gets lemma parts divided by '_' instead of spaces (second S.O.I. solved)
    lemma_synset = lemma+'_'+a[2]
    sentence = sentence.replace(anchor, lemma_synset)
    parts = sentence.split()            # gets the parts of the sentences
    index = parts.index(lemma_synset)   # gets the index of the sense in the list
    row = [lemma_synset]                # initializes the row with the center
    valid = 0
    pos = 0
    while valid < windowSize:           # for windowSize valid elements
        i = index-(pos+1)               # gets element before it
        if i<0:                         # if outside of the bound pad 
            row.insert(0, '<PAD>')
            valid += 1
        else:
            elem = parts[i]
            if isValid(elem):
                row.insert(0, elem)     # otherwise fill the list only if valid
                valid += 1
        pos +=1
    valid = 0
    pos = 0
    while valid < windowSize:           # same for the 'after' part
        i = index+(pos+1)
        if i>=len(parts):
            row.append('<PAD>')
            valid += 1
        else:
            elem = parts[i]
            if isValid(elem):
                row.append(elem)
                valid += 1
        pos +=1

    return row


# --- Function that creates the input tensor in the same structured-way of input_utils.py. ---
# :param sentences: 1D numpy array whose elements are the English sentences in the dataset
# :param annotations: 3D numpy array whose rows are arrays of length as nr. of annots for that sentence whose elems in turn are 3-vectors (anchor, lemma, id_synset)
# :param window_size: window size for the context
# :return inTensor: 2D numpy array whose row number is the conistent annotations one and the number of col. are 2*window_size + 1 (with all solvable S.O.I.s solved)

def fix_data(sentences, annotations, windowSize):

    inTensor = []                                   # contains the whole structure
    for i, annotation in enumerate(annotations):
        if annotation != []:
            sentence = sentences[i].lower()         # gets the lower-case version of the sentence (solved S.O.I.)
            for a in annotation:                    # a = (anchor, lemma, id_synset)
                anchor = a[0].lower()
                if ' '+anchor+' ' in sentence:      # if the annotation is valid
                    row = fix_row(sentence, a, windowSize)  # gets the row
                    inTensor.append(row)            # updates the structure
    return inTensor

        

