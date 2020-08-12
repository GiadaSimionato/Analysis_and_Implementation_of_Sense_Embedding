# -----------------------------------------------------------------------------------------------------------------------------------------------------------
#  Main script.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from utils import collect_bn2wn, trim_xml, filter_embedding
from sew_utils import parse_sew, getSewTensor, getSensesSew
from input_utils import get_tensor, get_map_senses
from analysis_inconsistencies import inconsistency_analysis
from fix_inconsistencies import fix_data
from score import score_model
from remove_limits import getNotBoundedInput

# Hyperparameters

WINDOW_SIZE = 4
EMBEDDING_SIZE = 200
NEGATIVE_SAMPLING = 15
EPOCHS = 5

# Paths

path_mapping = '../resources/bn2wn_mapping.txt'
path_savings = '../resources/embeddings.vec'
path_xml = '../EuroSense/eurosense.v1.0.high-precision.xml'
path_scoreData = '../combined.tab'
path_sew = '../sew_conservative'

print('Starting process...')
dict_bn2wn = collect_bn2wn(path_mapping)                    # dictionary whose keys are the BabelNet ids and values are the WordNet ones

print('Parsing EuroSense and Sew datasets...')
sentences, annotations = trim_xml(path_xml, dict_bn2wn)     # parses the EuroSense dataset
sentences = np.load('sentences.npy')
annotations = np.load('annotations.npy')
parse_sew(path_sew, dict_bn2wn)                             # parses the Sew dataset
sewSentences = np.load('sewSentences.npy')
sewAnnotations = np.load('sewAnnotations.npy')
print('Done')

print('Starting analysis of inconsistencies..')
inconsistency_analysis(sentences, annotations, dict_bn2wn)  # does the analysis of the inconsistencies
print('Done')

print('Collecting senses...')
word2senses = get_map_senses(annotations)                   # dictionary where the keys are the lemmas of the EuroSense dataset and the values are the lists of BabelNet ids
word2sensesSew = getSensesSew(sewAnnotations)               # dictionary where the keys are the lemmas of the Sew dataset and the values are the lists of BabelNet ids 
word2senses.update(word2sensesSew)                          # creates a common dictionary
print('Done')

print('Shaping tensors...')
in_tensor = fix_data(sentences, annotations, WINDOW_SIZE)            # creates the input tensor with EuroSense data
sewTensor = getSewTensor(sewSentences, sewAnnotations, WINDOW_SIZE)  # creates the input tensor with Sew data
in_tensor.extend(sewTensor)                                          # creates the common input tensor
print('Done')

print('Start training...')
model = Word2Vec(sentences=in_tensor, sg=0, size=EMBEDDING_SIZE, window=WINDOW_SIZE, min_count=1, workers=4, iter=EPOCHS)  # CBOW Gensim model
print('Done')

print('Start scoring...')
vw_table = model.wv
score = score_model(path_scoreData, vw_table, word2senses)  # computes the score
print('Done')
print('SCORE: ', score)

print('Saving model...')
model.wv.save_word2vec_format(path_savings)                 # saves the embeddings in the required format
filter_embedding(path_savings)                              # removes all the word embeddings 
print('Done process.')