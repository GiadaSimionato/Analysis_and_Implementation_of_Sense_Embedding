# -------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Utility methods for handling the Sew Conservative dataset.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It parses the Sew dataset returning sentences and annotations.
# It contains a method that builds the input tensor for the Sew dataset as the one implemented for the EuroSense and for retrieving the senses from the former.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import string
import numpy as np
from lxml import etree
from fix_inconsistences import isValid

# --- Function that parses a single XML file. ---
# :param path: the path of the XML file to parse
# :param bn2wn: a dictionary whose keys are the BabelNet ids and the values are the corr. WordNet ids
# :return text: the text of the XML file
# :return annot: the list of tuples like (BabelNet_id, mention, anchorStart, anchorEnd)

def trim_xml(path, bn2wn):

    annot = []      # will contain tuples like (BabelNet_id, mention, anchorStart, anchorEnd)
    content = etree.iterparse(path, events = ('start', ), remove_blank_text=True, encoding='UTF-8')  # creates the parsing structure
    for event, element in content:
        if element.tag == 'wikiArticle' and element.get('language').lower() == 'en':   # if the article is in English
            for child in element:
                if child.tag == 'text':
                    text = child.text.replace('\t', '').replace('\n', '')   # collects the text
                if child.tag == 'annotations':
                    for rec_child in child:   # for each annotation tag 
                        for elem in rec_child:
                            if elem.tag == 'babelNetID':
                                babelNetId = elem.text   # collects the babelNet id
                                if babelNetId not in bn2wn: # if not in WordNet 
                                    babelNetId = None       # sets it to None
                            if elem.tag == 'mention':
                                mention = elem.text      # collects the mention
                            if elem.tag == 'anchorStart':
                                anchorStart = elem.text  # collects the anchorStart
                            if elem.tag == 'anchorEnd':
                                anchorEnd = elem.text    # collects the anchorEnd
                        if babelNetId!=None and mention!=None and anchorStart!=None and anchorEnd!= None:  # if the tuple is valid
                            annot.append([babelNetId, mention.lower(), anchorStart, anchorEnd])     # appends the annotation
    return text, annot


# --- Function that parses the Sew dataset. ---
# :param path: path of the Sew dataset folder
# :param bn2wn: a dictionary whose keys are the BabelNet ids and the values are the corr. WordNet ids
# :return None: it saves the sentences and annotations

def parse_sew(path, bn2wn):

    texts = []      # contains all the texts of all the articles
    annots = []     # contains all the annotations
    list_folders = os.listdir(path)  # list of all the folders of the dataset
    for folder in list_folders:
        list_elements = os.listdir(path+'/'+folder)  # list of all the XML files in a specific folder
        for xmlFile in list_elements:
            path_xml = path + '/'+folder+'/'+xmlFile  
            if xmlFile != 'PaxHeader' and len(xmlFile)<80 and os.path.getsize(path_xml) < 92160:    # if the file can be handled
                try:
                    text, annot = trim_xml(path_xml, bn2wn)     # parse it
                    if text != None:                            # if the text is valid
                        texts.append(text)                      # collects the texts
                        annots.append(annot)                    # collects the annotations
                except Exception as e:                          # to handle invalid xmlChar values 
                    continue

    np.save('sewSentences', np.asarray(texts))
    np.save('sewAnnotations', np.asarray(annots))


# --- Function that builds the row of the input tensor. ---
# :param sentence: sentence
# :param a: (BabelNet_id, mention, anchorStart, anchorEnd)
# :param windowSize: window size
# :return row: a list of length 2*window_size + 1 centered in the lemma_synset

def getRow(sentence, a, windowSize):

    sentence = sentence.lower()
    anchorStart = int(a[2])
    anchorEnd = int(a[3])
    mention = a[1].replace(' ', '_').strip(string.punctuation)  # puts the mention in the correct form
    sense = mention + '_' + a[0]                                # builds the sense
    row = [sense]
    parts = sentence.split()                                    # splits the sentence
    valid = 0
    i = anchorStart-1
    while valid<windowSize:                                     # for windowSize times
        if i<0:
            row.insert(0, '<PAD>')                              # if out of bound then pad
            valid += 1
        else:
            elem = parts[i]
            if isValid(elem):
                row.insert(0, elem)                             # otherwise insert a valid element
                valid += 1
        i -= 1
    valid = 0
    i = anchorEnd
    while valid<windowSize:                                     # for windowSize times
        if i>=len(parts):                                       # if out of bound then pad
            row.append('<PAD>')
            valid += 1
        else:
            elem = parts[i]
            if isValid(elem):
                row.append(elem)                                # otherwise insert a valid element
                valid += 1
        i += 1
    return row


# --- Function that builds the input tensor. ---
# :param sentences: 1D numpy array whose elements are the English sentences in the dataset
# :param annotations: 3D numpy array whose rows are arrays of length as nr. of annots for that sentence whose elems in turn are 4-vectors (BabelNet_id, mention, anchorStart, anchorEnd)
# :param window_size: window size for the context
# :return tensor: 2D numpy array whose row number is the conistent annotations one and the number of col. are 2*window_size + 1

def getSewTensor(sentences, annotations, windowSize):

    tensor = []
    for i, annotation in enumerate(annotations):
        sentence = sentences[i]
        for a in annotation:  # a= (babelnet, mention, anchorStart, anchorEnd)
            try:
                row = getRow(sentence, a, windowSize)
                tensor.append(row)
            except Exception as e:
                continue
    return tensor


# --- Function that builds a dictionary whose keys are the lemmas and the values are lists of corresponding BabelNet ids. ---
# :param annotations: 3D numpy array whose rows are arrays of length as nr. of annots for that sentence whose elems in turn are 4-vectors (BabelNet_id, mention, anchorStart, anchorEnd)
# :return d: dictionary whose keys are the lemmas and the values are lists of corresponding BabelNet ids

def getSensesSew(annotations):

    d = dict()
    for annotation in annotations:
        for a in annotation:
            lemma = a[1].replace(' ', '_').strip(string.punctuation) # removes all the punctuation from the lemma and sets it in correct format
            bnId = a[0]
            if lemma in d and bnId not in d[lemma]:  # if the lemma is already in the dictionaty append the new id
                d[lemma].append(bnId)
            elif lemma not in d:                     # otherwise create new element with that id as list
                d[lemma] = [bnId]  
    return d
