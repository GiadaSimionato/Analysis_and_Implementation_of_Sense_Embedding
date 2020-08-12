# -----------------------------------------------------------------------------------------------------------------------------------------------------------
#   Principal Component Analysis and t-Distributed Stochastic Neighbor Embedding script.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It contains all the tools for performing the PCA analysis merged with the t-SNE technique of the model.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from utils import collect_bn2wn, trim_xml, create_vocabulary, reverse_vocabulary, doc2list, check, map_synsets
from input_utils import *
from gensim.models import Word2Vec, KeyedVectors
from sklearn.neighbors import NearestNeighbors
from score_notSkipping import score_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sew_utils import getSensesSew
from sklearn.preprocessing import StandardScaler

def get_senses(word, luTable, word2senses):
    emb_Set = []
    if word in word2senses:
        senses = word2senses[word]      # gets the list of all BabelNet ids of the lemma
        lemma_syns = []
        for sense in senses:
            lemma_syn = word+'_'+sense  # builds the sense       
            if lemma_syn in luTable:    # if there is an embedding of this sense retrieve it
                emb_Set.append(luTable[lemma_syn])
                lemma_syns.append(lemma_syn)
            else:
                emb_Set.append(np.zeros((1, 200)))
                lemma_syns.append('empty')
        assert len(emb_Set) == len(lemma_syns)
        return np.asarray(emb_Set), np.asarray(lemma_syns)     # converts the list as array
    else:
        return np.zeros((1,200)), np.array(['empty'])


path = '../combined.tab'

annotations = np.load('annotations.npy', allow_pickle=True)

new_annotations = np.load('sewAnnotations.npy', allow_pickle=True)


model = KeyedVectors.load_word2vec_format('./embeddings.vec', binary=False)

table = model.wv
word2senses = get_map_senses(annotations)

word2senses_sew = getSensesSew(new_annotations)

word2senses.update(word2senses_sew)


f = open(path, encoding='utf-8')
words = []
line = f.readline()  # discard first line (comments)
line = f.readline()
i = 2
while line!='':
    i += 1
    parts = line.split()
    if parts[0] not in words:
        words.append(parts[0])
    if parts[1] not in words:
        words.append(parts[1])
    line = f.readline()
f.close()


ext_senses = []
names = []
for word in words:
    sns, lemma_syns = get_senses(word, table, word2senses)
    for elem in sns:
        ext_senses.append(elem)
    for lmsyn in lemma_syns:
        names.append(lmsyn)
ext_senses = np.asarray(ext_senses)


senses = np.zeros((len(ext_senses),200))
for i in range(len(ext_senses)):
    senses[i, :] = ext_senses[i]
print(senses.shape)

pca = PCA(n_components=50)
pca.fit(senses)
comp_senses = pca.transform(senses)
tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
comp_senses = tsne.fit_transform(comp_senses)

nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(comp_senses)
sel_index = 40
distances, indices = nbrs.kneighbors(np.reshape(comp_senses[sel_index], (1, 3)))
print(indices)
print(len(names))
print('Sel. sense: ', names[sel_index])
for m in indices[0]:
    print(m)
    print('NN names', names[m])

for n in range(len(comp_senses)):
    if comp_senses[n].all() == np.zeros(3).all():
        comp_senses[n, :] = np.full(3, np.nan)

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
x, y, z = list(comp_senses[:, 0]), list(comp_senses[:, 1]), list(comp_senses[:, 2])

axis.scatter(x, y, z, s = 3, marker = ".")

mypoint = comp_senses[1]
dist = 0.0
max_point_index = 0
for q in range(len(comp_senses)):
    newpoint = comp_senses[q]
    newdist = np.linalg.norm(mypoint - newpoint)
    if newdist>dist:
        dist = newdist
        max_point_index = q
print(names[1])
print(names[max_point_index])


words_best = ['paper', 'journal', 'article']

_, first_senses = get_senses(words_best[0], table, word2senses)
_, second_senses = get_senses(words_best[1], table, word2senses)
_, third_senses = get_senses(words_best[2], table, word2senses)
print(first_senses)
print(second_senses)
print(third_senses)
# selected_names = [first_senses[0], second_senses[2], third_senses[0]]
selected_names = ['love_bn:00031470n', 'basketball_bn:00008890n']
cvec = ['red', 'green', 'black']
for name in selected_names:
    col_ind = selected_names.index(name)
    # Get the index of the name
    i = names.index(name)
    # Mark the labeled observations with a star marker
    axis.scatter(comp_senses[i,0], comp_senses[i,1], comp_senses[i,2],
                c=cvec[col_ind], marker='*', s=100)
axis.set_xlim3d(-10, 10)
axis.set_ylim3d(-10, 10)
axis.set_zlim3d(-10, 10)
plt.show()

