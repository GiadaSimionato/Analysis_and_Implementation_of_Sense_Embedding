# -----------------------------------------------------------------------------------------------------------------------------------------------------------
#   Methods that allow the building of the input tensor without the window-sized structure.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It provides the input tensor not shaped and not fixed, by replacing all the anchors with the corr. lemma_synset.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np

# --- Function that provides the row of the tensor. ---
# :param annotation: list of elements like (anchor, lemma, id_synset)
# :param sentence: sentence where annotations belong.
# :return list: list of elements of the replaced sentence 

def getRow(annotation, sentence): 

    sentence = sentence.replace(' ', '\t')
    for a in annotation:                                        # for each annotation
        if ' '+ a[0]+ ' ' in sentence:                          # if is a valid annotation
            anchor = a[0].replace(' ', '\t')
            lemma_synset = a[1]+'_'+a[2]                        # gets the sense
            sentence = sentence.replace(anchor, lemma_synset)   # replace the anchor with the sense
    return sentence.split()


# --- Function that provides the input tensor. ---
# :param sentences: 1D numpy array whose elements are the English sentences in the dataset
# :param annotations: 3D numpy array whose rows are arrays of length as nr. of annots for that sentence whose elems in turn are 3-vectors (anchor, lemma, id_synset)
# :return inTensor: array of sentences whose anchors are replaced by corr. senses

def getNotBoundedInput(sentences, annotations):

    inTensor = []
    for i, annotation in enumerate(annotations):        # for each list of annotations
        sentence = sentences[i]
        inTensor.append(getRow(annotation, sentence))   # gets the row and updates the tensor
    return np.asarray(inTensor)