# -----------------------------------------------------------------------------------------------------------------------------------------------------------
#   Methods to allow the manipulation of the results of the XML parsing so as to primarly shape them into tensors.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It builds the input tensor by shaping, according to the window size, the tensors collected from the XML parsing.
# It provides a dictionary so as to retrieve all the senses attached to a certain lemma.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

punctuation = ['.', ',', ':', ';', '"', "'", '!', '$', '£', '%', '&', '/', '(', ')', '=', '?', '^', '-', '_', '|', '<', '>', '+', '-', '*']

# --- Function that returns the row for the input of the model. ---
# :param sentence: the sentence to transform
# :param a: (anchor, lemma, id_synset)
# :param window_size: window size for the context
# :return out: a list of length 2*window_size + 1 centered in the lemma_synset

def get_partials(sentence, a, window_size):

    out = []                            # container for the input row parts
    before = []
    after = []
    pos = sentence.index(a[0])          # get the starting index of the anchor in the sentence
    p1 = sentence[:pos]                 # get the sub-sentence before the anchor
    p2 = sentence[pos+len(a[0]):]       # get the sub-sentence after the anchor
    for elem in p1.split():
        if elem not in punctuation:     # for every valid part of the 'before'
            before.append(elem)         # creates the list 
    for elem in p2.split():             
        if elem not in punctuation:     # for every valid part of the 'after'
            after.append(elem)          # creates the list
    new = a[1]+'_'+a[2]                 # creates the lemma_synset
    
    for i in range(len(before)-window_size, len(before)):   # backwords for window_size times updates the 2*window_size +1 list with:
        if i<0:
            out.append('<PAD>')         # a padding element if the window size requires more initial elements that available
        else:
            out.append(before[i])       # otherwise the element
    out.append(new)                     # add the sense to the list
    for i in range(window_size):        # repeats forward for window_size
        if i<len(after):
            out.append(after[i])
        else:
            out.append('<PAD>')

    return out
    

# --- Function that outputs the input tensor for the model. ---
# 
# :param sentences: 1D numpy array whose elements are the English sentences in the dataset
# :param annotations: 3D numpy array whose rows are arrays of length as nr. of annots for that sentence whose elems in turn are 3-vectors (anchor, lemma, id_synset)
# :param window_size: window size for the context
# :return tensor: 2D numpy array whose row number is the conistent annotations one and the number of col. are 2*window_size + 1

def get_tensor(sentences, annotations, window_size): 

    tensor = []     # input tensor

    for i, annotation in enumerate(annotations):    # annotation: list of tuples (anchor,lemma,id_syn) of sentence i
        for a in annotation:                        # a: (anchor, lemma, id_syn)
            if ' '+ a[0]+ ' ' in sentences[i]:      # if it is a consistent annotations
                partial = get_partials(sentences[i], a, window_size)    # retrives the row centered in this annotation
                tensor.append(partial)              # updates the tensor with the row
               
    return tensor


# --- Function that builds a dictionary whose keys are the lemmas and the values are lists of corresponding BabelNet ids. ---
# :param annotations: 3D numpy array whose rows are arrays of length as nr. of annots for that sentence whose elems in turn are 3-vectors (anchor, lemma, id_synset)
# :return d: dictionary whose keys are the lemmas and the values are lists of corresponding BabelNet ids

def get_map_senses(annotations):

    d = dict()
    for annotation in annotations:
        for a in annotation:                        # a = (anchor,lemma,id_synset)
            lemma = a[1]                            # gets the lemma
            if lemma in d and a[2] not in d[lemma]: # if the lemma is already in the dictionaty append the new id
                d[lemma].append(a[2])
            elif lemma not in d:
                d[lemma] = [a[2]]                   # otherwise create new element with that id as list
    return d
