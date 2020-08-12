# -----------------------------------------------------------------------------------------------------------------------------------------------------------
#   Utility methods to provide the required resources for the input and output of the main script.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It parses the EuroSense dataset returning the English sentences and annotations with a bi-univocal correspondance, while creating a dictionary of 
# babelNet-WordNet ids relations as to fulfill this purpose.
# It contains a method that builds a document containing all lemma_synsets' embeddings in the required format and path.
# Additional methods are contained: used to confirm that the scoring file contains words not present in the annotation file; alternative implementations 
# of the vocabulary building so as to bypass the fact that Gensim automatically discards all the words that appear less than three times by providing
# the possibility to change this threshold.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

from lxml import etree
import numpy as np

punctuation = ['.', ',', ':', ';', '"', "'", '!', '$', '£', '%', '&', '/', '(', ')', '=', '?', '^', '-', '_', '|', '<', '>', '+', '-', '*']

# --- Function that builds the dictionary of BabelNet-WordNet ids correspondances. ---
# :param path: path of the bn2wn_mapping file
# :return d: dictionary whose keys are the BabelNet ids and the values are the WordNet ones

def collect_bn2wn(path):

    d = dict()
    txt = open(path, encoding='utf-8')
    line = txt.readline()
    while line!='':
        line = line.strip()
        partials = line.split('\t')     # parses the line
        d[partials[0]] = partials[1]    # builds the key as the BabelNet id with corresponding WordNet id
        line = txt.readline()
    txt.close()
    return d

# --- Function that parse the EuroSense dataset. ---
# :param path: the path of the EuroSense XML file to parse
# :param d: dictionary of the BabelNet-WordNet ids correspondances
# :return sentences: list of English sentences encountered while parsing the XML file
# :return annotations: list of the same length of sentences whose elements are lists of the corr. annots with elems in the form of triples (anchor, lemma, id_synset)

def trim_xml(path, d):
    
    sentences = []
    annotations = []
    content = etree.iterparse(path, remove_blank_text=True, tag = 'sentence')   # build the tree for parsing according to the tag 'sentence'
    for event, element in content:                                              # for each 'sentence'
        partial_annotations = []                                                # list of annotations for this sentence
        for child in element:
            if child.tag == 'text' and child.get('lang')=="en":
                line = child.text
                sentences.append(line)                                          # collects the English sentence from the text tag
            if child.tag == 'annotations':
                for rec_child in child:                                         # for each English annotation tag
                    if rec_child.get('lang') == 'en':   
                        bn_id = rec_child.text                                  # gets BabelNet id
                        if bn_id in d:                                          # if there is a corresp. with WordNet
                            anchor = rec_child.get('anchor')                    # gets anchor
                            lemma = rec_child.get('lemma')                      # gets lemma
                            partial_annotations.append([anchor,lemma,bn_id])    
        annotations.append(partial_annotations)                                 # updates common tensor
        element.clear()                                                         # discard the subtree freeing the allocated memory
    return sentences,  annotations          

# --- Function that filters the embedding.vec file by overriding it with only sense embeddings. ---
# :param path: path of the embeddings.vec file in the KeyedVector format
# :return None: it writes in the path file the sense embeddings in the KeyedVector format required

def filter_embedding(path):

    dst_path = '../resources/full_embeddings.vec'       # path for a temporary file
    f = open(path, encoding='utf-8')
    dst_file = open(dst_path, 'w', encoding='utf-8')
    embSize = f.readline().split()[1]                   # gets the embedding size from the first row
    count = 0                                           # keeps track of the number of sense embeddings
    line = f.readline()
    while line != '':
        if '_bn:' in line:                              # if it is a sense then write it in the document and increment the counter
            dst_file.write(line)
            count += 1
        line = f.readline()
    f.close()
    dst_file.close()
    string = str(count) + ' ' + str(embSize) + '\n'     # builds the first row as the nr of sense emb. and their emb. size
    
    file_src = open(dst_path, encoding='utf-8')         
    file_dst = open(path, 'w', encoding='utf-8')        # overrides the embedding.vec file
    file_dst.write(string)                              # writes the firse line
    for line in file_src:                               # copies the lines from the temp. file to the embedding.vec one
        file_dst.write(line)
    file_src.close()
    file_dst.close()

# --------- ADDITIONAL METHODS --------------------------------------------------------------------------------

# --- Function that counts the occurrences of each word in the text. ---
# :param dictionary: the latest version of the dictionary whose keys are the words and their values the corr. frequency
# :param sentence: the text to add to the vocabulary
# :return dictionary: the updated version of the structured dictionary
# :return i: the amount of valid words encountered in sentence

def count_occurrences(dictionary, sentence):

    i = 0                                   # total occurrences counter
    parts = sentence.split()  
    for part in parts:                      # for each word of the sentence
        if part not in punctuation:         # discards the punctuation symbols
            i += 1
            if part not in dictionary:
                dictionary[part] = 1        # if the word is new add to the dict. with freq. 1       
            else:
                dictionary[part] = dictionary[part]+1 # ptherwise increment the freq.
    return dictionary, i


# --- Function that creates the vocabulary from the plain text according to a threshold. ---
# :param sentences: the list of the plain text
# :param threshold: the threshold for the relative frequences of the words under which the words are discarded
# :return dictionary: dictionary whose keys are the words and their values are the indices of those words

def create_vocabulary(sentences, threshold):

    index = 0
    freq_dict = dict()
    dictionary = {'<PAD>': 0, '<UNK>': 1}                       # augments the dictionary with the unknown and padding words

    for elem in sentences:                                      # for each word
        if elem != None:
            freq_dict, i = count_occurrences(freq_dict, elem)    # gets the updated dictionary with the frequences and the number of total occ. of that word
            index += i

    for word, freq in freq_dict.items():                        
        if freq/index > threshold:                              # converts the frequences into relative ones and filters those that are under the threshold
            dictionary[word] = len(dictionary)                  # updates their values with a unique and progr. index
    return dictionary


# --- Function that inverts the keys and values of a dictionary. ---
# :param d: the dictionary to invert
# :return d: dictionary with key and values swapped.

def reverse_vocabulary(d):

    return {v:k for k,v in d.items()}


# --- Function that checks if words are contained in a dictionary (used for confirm that the combined.tab file contained words not seen in annots). ---
# :param map_syn: dictionary
# :param word_list: list of words
# :return boolean: True if every element of the list is contained in the dictionary, false otherwise

def check(map_syn, word_list):
    for elem in word_list:
        if elem not in map_syn:
            return False
    return True


# --- Function that transforms a file (used for combined.tab) into a list of words. ---
# :param path: the path of the file to convert
# :return l: the list of words

def doc2list(path):
    
    l = []
    f = open(path, encoding='utf-8')
    line = f.readline()
    line = f.readline()
    while line!='':
        parts = line.split()
        l.append(parts[0])
        l.append(parts[1])
        line = f.readline()
    f.close()
    return l


    

   