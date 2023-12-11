'''
Assignment No: 02
Name: Yash Pramod Dhage
Roll: 70
Batch: B4
Title: Assignment to implement  bag-of-words approach tf-idf on data. Create embedding using
        Word2Vec using Gensim python library.
'''

from gensim.utils import simple_preprocess
from gensim import corpora, models
import numpy as np

# Bag-of-Words

#text2 = open('sample_text.txt', encoding='utf-8')

text2 = ["""natural language processing (NLP) is a field of artificial intelligence that focuses on the 
            interaction between computers and humans through natural language. It enables machines to understand,
            interpret, and generate human language in a valuable way."""]

tokens2 = []
# for line in text2.read().split('.'):
for line in text2[0].split('.'):
    tokens2.append(simple_preprocess(line, deacc=True))

g_dict2 = corpora.Dictionary(tokens2)

print("The dictionary has: " + str(len(g_dict2)) + " tokens")
print(g_dict2.token2id)
print("\n")

g_bow =[g_dict2.doc2bow(token, allow_update = True) for token in tokens2]
print("Bag of Words : ", g_bow)



#Term Frequency – Inverse Document Frequency (TF-ID)

text = [
        "The weather today is sunny and warm, perfect for outdoor activities.",
        "I have a lot of work to do, and the deadline is approaching fast.",
        "I enjoy reading books in my free time, especially science fiction novels."
        ]

g_dict = corpora.Dictionary([simple_preprocess(line) for line in text])
g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in text]

print("\nTerm Frequency – Inverse Document Frequency (TF-ID)")
print("Dictionary : ")
for item in g_bow:
    print([[g_dict[id], freq] for id, freq in item])

g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')

print("\n TF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])



#word2vec

from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count

print("\n Word2vec \n")

data = [
    "this is a sentence",
    "another example sentence",
    "word embeddings are useful",
    "word2vec is a popular model",
]

# Tokenize the text data (split sentences into words)
tokenized_data = [sentence.split() for sentence in data]

# Create a Word2Vec model
w2v_model = Word2Vec(tokenized_data, min_count=0, workers=cpu_count())

# Find the most similar words to 'word'
similar_words = w2v_model.wv.most_similar('word')

for word, score in similar_words:
    print(f"{word}: {score}")



'''
OUTPUT:

The dictionary has: 30 tokens
{'and': 0, 'artificial': 1, 'between': 2, 'computers': 3, 'field': 4, 'focuses': 5, 'humans': 6, 'intelligence': 7, 'interaction': 8, 'is': 9, 'language': 10, 'natural': 11, 'nlp': 12, 'of': 13, 'on': 14, 'processing': 15, 'that': 16, 'the': 17, 'through': 18, 'enables': 19, 'generate': 20, 'human': 21, 'in': 22, 'interpret': 23, 'it': 24, 'machines': 25, 'to': 26, 'understand': 27, 'valuable': 28, 'way': 29}

Bag of Words :  [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 2), (11, 2), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1)], [(0, 1), (10, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 1)], []]

Term Frequency – Inverse Document Frequency (TF-ID)
Dictionary : 
[['activities', 1], ['and', 1], ['for', 1], ['is', 1], ['outdoor', 1], ['perfect', 1], ['sunny', 1], ['the', 1], ['today', 1], ['warm', 1], ['weather', 1]]
[['and', 1], ['is', 1], ['the', 1], ['approaching', 1], ['deadline', 1], ['do', 1], ['fast', 1], ['have', 1], ['lot', 1], ['of', 1], ['to', 1], ['work', 1]]
[['books', 1], ['enjoy', 1], ['especially', 1], ['fiction', 1], ['free', 1], ['in', 1], ['my', 1], ['novels', 1], ['reading', 1], ['science', 1], ['time', 1]]

 TF-IDF Vector:
[['activities', 0.34], ['and', 0.17], ['for', 0.34], ['is', 0.17], ['outdoor', 0.34], ['perfect', 0.34], ['sunny', 0.34], ['the', 0.17], ['today', 0.34], ['warm', 0.34], ['weather', 0.34]]
[['and', 0.16], ['is', 0.16], ['the', 0.16], ['approaching', 0.32], ['deadline', 0.32], ['do', 0.32], ['fast', 0.32], ['have', 0.32], ['lot', 0.32], ['of', 0.32], ['to', 0.32], ['work', 0.32]]
[['books', 0.3], ['enjoy', 0.3], ['especially', 0.3], ['fiction', 0.3], ['free', 0.3], ['in', 0.3], ['my', 0.3], ['novels', 0.3], ['reading', 0.3], ['science', 0.3], ['time', 0.3]]


Word2vec

sentence: 0.21617141366004944
embeddings: 0.044689226895570755
example: 0.015025208704173565
useful: 0.0019510718993842602
is: -0.03284316137433052
this: -0.04568909481167793
another: -0.0742427185177803
are: -0.09326908737421036
a: -0.09575342386960983
word2vec: -0.10513807833194733


'''

# Reference: https://www.analyticsvidhya.com/blog/2022/03/learn-basics-of-natural-language-processing-nlp-using-gensim-part-1/



# Bag of Words (BoW):

# Definition: Bag of Words is a simple and widely used technique in natural language processing for text representation. It represents a document as an unordered set of words, disregarding grammar and word order but keeping track of word frequency.

# Process:

# Create a vocabulary: Compile a list of unique words from the entire corpus.
# Create vectors: Represent each document as a vector, where each element corresponds to the frequency of a word from the vocabulary.
# Example:

# Consider two documents: "I love natural language processing" and "Natural language processing is fascinating."
# Vocabulary: ["I", "love", "natural", "language", "processing", "is", "fascinating"]
# Document vectors: [1, 1, 1, 1, 1, 0, 0] and [0, 0, 2, 1, 1, 1, 1]
# Term Frequency-Inverse Document Frequency (TF-IDF):

# Definition: TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus).

# Process:

# Calculate Term Frequency (TF): Number of times a word appears in a document divided by the total number of words in that document.
# Calculate Inverse Document Frequency (IDF): Logarithm of the total number of documents divided by the number of documents containing the word.
# Multiply TF by IDF to get the TF-IDF score for each word in a document.
# Example:

# For the document "I love natural language processing," the TF-IDF scores might emphasize the importance of "love" and "processing" over common words like "I" and "natural."
# Word2Vec:

# Definition: Word2Vec is a word embedding technique that represents words as dense vectors in a continuous vector space. It is based on the idea that words with similar meanings should have similar representations.

# Process:

# Word2Vec models are trained on large corpora using neural networks. The models learn to predict the context of a word (Skip-gram) or predict a word given its context (Continuous Bag of Words, CBOW).
# The learned vectors capture semantic relationships, and words with similar meanings are closer in the vector space.
# Example:

# After training a Word2Vec model, words like "king" and "queen" may have vectors that are close in the vector space, reflecting their semantic similarity.
# In summary, Bag of Words is a simple representation based on word frequency, TF-IDF considers the importance of words in a document relative to a corpus, and Word2Vec provides dense vector representations capturing semantic relationships between words. Each method has its strengths and weaknesses, and the choice depends on the specific requirements of the NLP task.