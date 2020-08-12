# -----------------------------------------------------------------------------------------------------------------------------------------------------------
#   Inconsistencies analysis of the dataset
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It analyses all the sources of inconsistencies found in trimming the dataset with the purpose of filtering wrong data as to improve sense embedding while
# solving some inconsistencies as to augment the training corpus.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

from langdetect import detect_langs
from nltk.corpus import wordnet as wn
import nltk

# --- Function that detects the percentage of annotations whose anchors are not in the corresponding sentences. ---
# :param sentences: a 1D numpy array of all the English sentences collected from the dataset
# :param annotations: a 3D numpy array with nr. of rows the nr. of sentences, nr. of cols the nr. of annotations for each sentence and as 3rd dim. a 3-element vector (anchor, lemma, id_synset)
# :return p: percentage of annotations whose anchors are not in the corresponding sentences

def not_in_sentence(sentences, annotations):

    tot = 0                                 # total collected annotations 
    positives = 0                           # nr. of annotations whose anchors are not in the sentences.

    for i, annotation in enumerate(annotations):
        sentence = sentences[i]             # get the corresponding sentence
        for a in annotation:                # a = (anchor,lemma,id_synset)
            string = ' '+a[0]+' '           # pad the anchor with spaces to detect if it is in the sentence as a single word
            if string not in sentence:      # if such anchor is not in the corresponding sentence, increment the counter
                positives+=1
            tot+=1

    return 100*positives/tot                # compute the percentage


# --- Function that detects the percentage of annotations whose anchors are not in the corr. sentence as single words but are anyway part of it. ---
# :param sentences: a 1D numpy array of all the English sentences collected from the dataset
# :param annotations: a 3D numpy array with nr. of rows the nr. of sentences, nr. of cols the nr. of annotations for each sentence and as 3rd dim. a 3-element vector (anchor, lemma, id_synset)
# :return p: percentage of annotations whose anchors are not in the corr. sentence as single words but are anyway part of it

def not_in_but_partial(sentences, annotations):

    tot = 0                 # total non-consistent annotations 
    positives = 0           # total non-consistent annotations but anyway part of the sentence 

    for i, annotation in enumerate(annotations):
        sentence = sentences[i]                         # get the corresponding sentence
        for a in annotation:                            #a = (anchor,lemma,id)
            string = ' '+a[0]+' '                       # pad the anchor with spaces to detect if it is in the sentence as a single word
            if string not in sentence:                  # if inconsitent 
                if a[0].lower() in sentence.lower():    # if it's part of the sentence
                    positives += 1                      # increment the counter
                tot+=1                                  

    return 100*positives/tot                            # compute the percentage


# --- Function that detects the percentage of annotations whose anchors are not in the corr. sentence due to an upper-lower mismatch. ---
# :param sentences: a 1D numpy array of all the English sentences collected from the dataset
# :param annotations: a 3D numpy array with nr. of rows the nr. of sentences, nr. of cols the nr. of annotations for each sentence and as 3rd dim. a 3-element vector (anchor, lemma, id_synset)
# :return p: percentage of annotations whose anchors are not in the corr. sentence due to an upper-lower mismatch

def not_in_low_up(sentences, annotations):

    tot = 0                 # total non-consistent annotations 
    positives = 0           # total non-consistent annotations due to upper-lower mismatch

    for i, annotation in enumerate(annotations):
        sentence = sentences[i]                         # get the corresponding sentence
        for a in annotation:                            # a = (anchor,lemma,id)
            string = ' '+a[0]+' '                       # pad the anchor with spaces to detect if it is in the sentence as a single word
            if string not in sentence:                  # if inconsitent
                if string.lower() in sentence.lower():  # if the padded version is part of the sentence
                    positives += 1                      # increment the counter
                tot+=1

    return 100*positives/tot                            # compute the percentage


# --- Function that detects whether the language of the anchor is English ---
# :param anchor: a string
# :return boolean: True if the language detected is English, False otherwise

def isEnglish(anchor):

    try:
        langs = detect_langs(anchor)        # returns a list of elems with format xx:prob where xx is the abbreviation of the language and probs is the probability that belongs to that lang.
        return str(langs[0])[:2] == 'en'    # returns whether is English or not
    except:                                 # if not possible to estimate language (like numbers, acronymus etc.)
        return True                         # suppose it's English


# --- Function that detects the percentage of annotations whose anchors are not in the corr. sentence and belong to another language. ---
# :param sentences: a 1D numpy array of all the English sentences collected from the dataset
# :param annotations: a 3D numpy array with nr. of rows the nr. of sentences, nr. of cols the nr. of annotations for each sentence and as 3rd dim. a 3-element vector (anchor, lemma, id_synset)
# :return p: percentage of annotations whose anchors are not in the corr. sentence and belong to another language

def not_in_other_lang(sentences, annotations):

    tot = 0
    positives = 0

    for i, annotation in enumerate(annotations):
        sentence = sentences[i]             # get the corresponding sentence
        for a in annotation:                # a = (anchor,lemma,id)
            string = ' '+a[0]+' '           # pad the anchor with spaces to detect if it is in the sentence as a single word
            if string not in sentence:      # if inconsitent
                if not isEnglish(a[0]):     # if not in English
                    positives += 1          # increment the counter
                tot+=1

    return 100*positives/tot                # compute the percentage


# --- Function that detects the percentage of consistent annotations but whose synsets don't correspond to the lemmas. ---
# :param sentences: a 1D numpy array of all the English sentences collected from the dataset
# :param annotations: a 3D numpy array with nr. of rows the nr. of sentences, nr. of cols the nr. of annotations for each sentence and as 3rd dim. a 3-element vector (anchor, lemma, id_synset)
# :param bnId2wnId: dictionary that maps the BabelNet ids to WordNet ones
# :return p: percentage of consistent annotations but whose synsets don't correspond to the lemmas

def in_but_wrong_annot(sentences, annotations, bnId2wnId):

    tot = 0
    positives = 0

    for i, annotation in enumerate(annotations):
        sentence = sentences[i]             # get the corresponding sentence
        for a in annotation:                # a = (anchor,lemma,id)
            string = ' '+a[0]+' '           # pad the anchor with spaces to detect if it is in the sentence as a single word
            if string in sentence:          # if consistent
                tot += 1
                offset = bnId2wnId[a[2]]    # get the corr. WordNet id
                synset = wn.synset_from_pos_and_offset( offset[-1], int(offset[:-1]))   # get the synset
                lemma = str(synset)[8:-7]   # extract the sense from the synset
                if lemma != a[1]:           # if it is different from the corr. lemma
                    positives += 1          # increment the counter

    return 100*positives/tot                # compute the percentage


# --- Function that detects the percentage of wrong-associated-to-synset annotations due to upper-lower mismatch. ---
# :param sentences: a 1D numpy array of all the English sentences collected from the dataset
# :param annotations: a 3D numpy array with nr. of rows the nr. of sentences, nr. of cols the nr. of annotations for each sentence and as 3rd dim. a 3-element vector (anchor, lemma, id_synset)
# :param bnId2wnId: dictionary that maps the BabelNet ids to WordNet ones
# :return p: percentage of wrong-associated-to-synset annotations due to upper-lower mismatch

def in_but_up_low_mismatch(sentences, annotations, bnId2wnId):

    tot = 0
    positives = 0

    for i, annotation in enumerate(annotations):
        sentence = sentences[i]
        for a in annotation:         # a = (anchor,lemma,id)
            string = ' '+a[0]+' '
            if string in sentence:   # if consistent
                tot += 1
                offset = bnId2wnId[a[2]]
                synset = wn.synset_from_pos_and_offset( offset[-1], int(offset[:-1]))
                lemma = str(synset)[8:-7]
                if lemma != a[1]:                       # if wrong-associated 
                    if lemma.lower() == a[1].lower():   # but with a lower reduction equivalent
                        positives += 1

    return 100*positives/tot    # compute the percentage


# --- Function that edits the string by replacing the space with undercores. ---
# :param string: string to edit
# :return string: string with spaces replaced with underscores

def edit_string(string):

    return string.replace(' ', '_')


# --- Function that detects the percentage of wrong-associated-to-synset annotations due to missing underscores among words of lemmas. ---
# :param sentences: a 1D numpy array of all the English sentences collected from the dataset
# :param annotations: a 3D numpy array with nr. of rows the nr. of sentences, nr. of cols the nr. of annotations for each sentence and as 3rd dim. a 3-element vector (anchor, lemma, id_synset)
# :param bnId2wnId: dictionary that maps the BabelNet ids to WordNet ones
# :return p: percentage of wrong-associated-to-synset annotations due to missing underscores among words of lemmas

def in_but_underscore_mismatch(sentences, annotations, bnId2wnId):

    tot = 0
    positives = 0

    for i, annotation in enumerate(annotations):
        sentence = sentences[i]
        for a in annotation:            # a = (anchor,lemma,id)
            string = ' '+a[0]+' '
            if string in sentence:      # if consistent
                tot += 1
                offset = bnId2wnId[a[2]]
                synset = wn.synset_from_pos_and_offset( offset[-1], int(offset[:-1]))
                lemma = str(synset)[8:-7]
                if lemma != a[1]:      # if wrong-associated
                    if lemma.lower() == edit_string(a[1].lower()): # if equivalent with underscored version
                        positives += 1

    return 100*positives/tot   # compute the percentage


# --- Function that collects all the senses of all the annotations of a specific sentence. ---
# :param annotation: list of tuples (anchor, lemma, id_synset) of a specific sentence
# :return l: list of senses drawn from synset ids

def get_lemmas(annotation):

    l = []
    for elem in annotation:
        l.append(elem[1].lower())
    return l


# --- Function that detects the percentage of wrong-associated-to-synset annotations due to a shift in the corr. sentences. ---
# :param sentences: a 1D numpy array of all the English sentences collected from the dataset
# :param annotations: a 3D numpy array with nr. of rows the nr. of sentences, nr. of cols the nr. of annotations for each sentence and as 3rd dim. a 3-element vector (anchor, lemma, id_synset)
# :param bnId2wnId: dictionary that maps the BabelNet ids to WordNet ones
# :return p: percentage of wrong-associated-to-synset annotations due to a shift in the corr. sentences

def in_wrong_but_shifted(sentences, annotations, bnId2wnId):

    tot = 0
    positives = 0

    for i, annotation in enumerate(annotations):
        sentence = sentences[i]
        for a in annotation:         # a = (anchor,lemma,id)
            string = ' '+a[0]+' '
            if string in sentence:   # if consistent
                tot += 1
                offset = bnId2wnId[a[2]]
                synset = wn.synset_from_pos_and_offset( offset[-1], int(offset[:-1]))
                lemma = str(synset)[8:-7]
                if lemma.lower() != a[1].lower():   # if wrong-associated
                    if lemma.lower() in get_lemmas(annotation):  # if lemma in other synsets of the same sentence
                        positives += 1

    return 100*positives/tot     # compute the percentage

# -------- ADDITIONAL METHODS -------------------------------------------------------------
# Functions that allow a more deep analysis of the language incongruencies

# --- Function that returns the abbreviation of the language of the input word ---
# :param anchor: string whose language has to be detected
# :return xx: abbreviation of the language

def which_language(anchor):

    try:
        langs = detect_langs(anchor)
        return str(langs[0])[:2]
    except:     # if not possible to estimate language (like numbers, acronymus etc.) suppose English
        return 'en'


# --- Function that collects the list of all the anchors' languages of the annotations. ---
# :param annotations: a 3D numpy array with nr. of rows the nr. of sentences, nr. of cols the nr. of annotations for each sentence and as 3rd dim. a 3-element vector (anchor, lemma, id_synset)
# :return langs: list of all the anchors' languages

def collect_languages(annotations):

    langs = []

    for annotation in annotations:
        l = []  # collects all the 
        for a in annotation:     # a = (anchor,lemma,id)
            l.append(which_language(a[0])) 
        langs.append(l)

    return langs

 # -------------------------------------------------------------------------------

# --- Main function that executes all the functions descibed above and prints the results. ---

def inconsistency_analysis(sentences, annotations, bnId2wnId):

    print('Starting analysis...')

    result = not_in_sentence(sentences, annotations)
    print('There is a {} \%\ of the annotations whose anchors are not in the corresponding sentence.'.format(result))
    
    result = not_in_but_partial(sentences, annotations)
    print('There is a {} \%\ of the not-consistent annotations whose anchors are anyway part of the corresponding sentence.'.format(result))

    result = not_in_low_up(sentences, annotations)
    print('There is a {} \%\ of the not-consistent annotations whose not-validity is due to an upper-lower format mismatch.'.format(result))

    result = not_in_other_lang(sentences, annotations)
    print('There is a {} \%\ of the not-consistent annotations whose anchors belong to other languages.'.format(result))

    result = in_but_wrong_annot(sentences, annotations, bnId2wnId)
    print('There is a {} \%\ of the consistent annotations whose synsets do not match with the corresponding lemma.'.format(result))
    
    result = in_but_up_low_mismatch(sentences, annotations, bnId2wnId)
    print('There is a {} \%\ of the consistent annotations whose mismatch between synsets and corresponding lemma is due an upper-lower format mismatch.'.format(result))

    result = in_but_underscore_mismatch(sentences, annotations, bnId2wnId)
    print('There is a {} \%\ of the consistent annotations whose mismatch is due a missing underscore among multiple-words lemmas.'.format(result))

    result = in_wrong_but_shifted(sentences, annotations, bnId2wnId)
    print('There is a {} \%\ of the consistent annotations whose mismatch is due to a shift of annotations into the same sentence.'.format(result))

    return