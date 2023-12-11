#Title: "Implementation of Bag of Words using Gensim"
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora
inp = [""" . """]
tokens = []
for line in inp[0].split('.'):
    tokens.append(simple_preprocess(line, deacc=True))
g_dict = corpora.Dictionary(tokens)
print("The dictionary has: " + str(len(g_dict)) + " tokens")
print(g_dict.token2id)
print("\n")
bow =[g_dict.doc2bow(t, allow_update = True) for t in tokens]
print("Bag of Words : ", bow)
#======================================================================

import pprint
from gensim import models
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

g_tfidf = models.TfidfModel(bow, smartirs='ntc')

for item in g_tfidf[bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])

#BoW and TF-IDF are techniques that help us convert text sentences into numeric vectors.Term Frequency-Inverse Document Frequency.
    dis:If the new sentences contain new words, then our vocabulary size would increase and thereby, the length of the vectors would increase too.
    Term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.
    It is a measure of how frequently a term, t, appears in a document, d:
    iDF is a measure of how important a term is.add()TF-IDF(‘this’, Review 2) = TF(‘this’, Review 2) * IDF(‘this’) = 1/8 * 0 = 0
#tf(t,d) = count of t in d / number of words in d
#df(t) = occurrence of t in sentences
idf(t) = log(Number of sentences/ df(t))                                                                                                      
tf-idf(t, d) = tf(t, d) * idf(t)