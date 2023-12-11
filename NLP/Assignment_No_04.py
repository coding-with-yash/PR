'''
Assignment No: 04
Name: Yash Pramod Dhage
Roll: 70
Batch: B4
Title: Implement Bi-gram, Tri-gram word sequence and its count in text input data using NLTK library.
'''

from nltk import ngrams

from nltk.util import ngrams
#unigram model
n = 1
sentence = 'While unigram model sentences will only exclude the UNK token, models will also exclude all other words already in the sentence.NTK provides another function everygrams that converts a sentence into unigram, bigram, trigram, and so on till the ngrams, where n is the length of the sentence. In short, this function generates ngrams for all possible values of n.'
unigrams = ngrams(sentence.split(), n)

for item in unigrams:
    print(item)
#bigram model
n = 2
sentence = 'While unigram model sentences will only exclude the UNK token, models will also exclude all other words already in the sentence.NTK provides another function everygrams that converts a sentence into unigram, bigram, trigram, and so on till the ngrams, where n is the length of the sentence. In short, this function generates ngrams for all possible values of n.'
unigrams = ngrams(sentence.split(), n)

for item in unigrams:
    print(item)
#trigram model
n = 3
sentence = 'While unigram model sentences will only exclude the UNK token, models will also exclude all other words already in the sentence.NTK provides another function everygrams that converts a sentence into unigram, bigram, trigram, and so on till the ngrams, where n is the length of the sentence. In short, this function generates ngrams for all possible values of n.'
unigrams = ngrams(sentence.split(), n)

for item in unigrams:
    print(item)

#using text file input
from nltk import ngrams
# file = open("/home/exam/Desktop/NLP_LAB75/al.txt")
file = open("D:/Eng_4_Sem_7/Natural Language Processing/Lab_Practical/NLP_LAB/sample_text.txt")
for i in file.readlines():
    cumulative = i
    sentences = i.split(".")
    counter = 0
    for sentence in sentences:
        print("For sentence", counter + 1, ", trigrams are: ")
        trigrams = ngrams(sentence.split(" "), 3)
        for grams in trigrams:
            print(grams)
        counter += 1
        print()

'''
output

('While',)
('unigram',)
('model',)
('sentences',)
('will',)
('only',)
('exclude',)
('the',)
('UNK',)
('token,',)
('models',)
('will',)
('also',)
('exclude',)
('all',)
('other',)
('words',)
('already',)
('in',)
('the',)
('sentence.NTK',)
('provides',)
('another',)
('function',)
('everygrams',)
('that',)
('converts',)
('a',)
('sentence',)
('into',)
('unigram,',)
('bigram,',)
('trigram,',)
('and',)
('so',)
('on',)
('till',)
('the',)
('ngrams,',)
('where',)
('n',)
('is',)
('the',)
('length',)
('of',)
('the',)
('sentence.',)
('In',)
('short,',)
('this',)
('function',)
('generates',)
('ngrams',)
('for',)
('all',)
('possible',)
('values',)
('of',)
('n.',)
('While', 'unigram')
('unigram', 'model')
('model', 'sentences')
('sentences', 'will')
('will', 'only')
('only', 'exclude')
('exclude', 'the')
('the', 'UNK')
('UNK', 'token,')
('token,', 'models')
('models', 'will')
('will', 'also')
('also', 'exclude')
('exclude', 'all')
('all', 'other')
('other', 'words')
('words', 'already')
('already', 'in')
('in', 'the')
('the', 'sentence.NTK')
('sentence.NTK', 'provides')
('provides', 'another')
('another', 'function')
('function', 'everygrams')
('everygrams', 'that')
('that', 'converts')
('converts', 'a')
('a', 'sentence')
('sentence', 'into')
('into', 'unigram,')
('unigram,', 'bigram,')
('bigram,', 'trigram,')
('trigram,', 'and')
('and', 'so')
('so', 'on')
('on', 'till')
('till', 'the')
('the', 'ngrams,')
('ngrams,', 'where')
('where', 'n')
('n', 'is')
('is', 'the')
('the', 'length')
('length', 'of')
('of', 'the')
('the', 'sentence.')
('sentence.', 'In')
('In', 'short,')
('short,', 'this')
('this', 'function')
('function', 'generates')
('generates', 'ngrams')
('ngrams', 'for')
('for', 'all')
('all', 'possible')
('possible', 'values')
('values', 'of')
('of', 'n.')
('While', 'unigram', 'model')
('unigram', 'model', 'sentences')
('model', 'sentences', 'will')
('sentences', 'will', 'only')
('will', 'only', 'exclude')
('only', 'exclude', 'the')
('exclude', 'the', 'UNK')
('the', 'UNK', 'token,')
('UNK', 'token,', 'models')
('token,', 'models', 'will')
('models', 'will', 'also')
('will', 'also', 'exclude')
('also', 'exclude', 'all')
('exclude', 'all', 'other')
('all', 'other', 'words')
('other', 'words', 'already')
('words', 'already', 'in')
('already', 'in', 'the')
('in', 'the', 'sentence.NTK')
('the', 'sentence.NTK', 'provides')
('sentence.NTK', 'provides', 'another')
('provides', 'another', 'function')
('another', 'function', 'everygrams')
('function', 'everygrams', 'that')
('everygrams', 'that', 'converts')
('that', 'converts', 'a')
('converts', 'a', 'sentence')
('a', 'sentence', 'into')
('sentence', 'into', 'unigram,')
('into', 'unigram,', 'bigram,')
('unigram,', 'bigram,', 'trigram,')
('bigram,', 'trigram,', 'and')
('trigram,', 'and', 'so')
('and', 'so', 'on')
('so', 'on', 'till')
('on', 'till', 'the')
('till', 'the', 'ngrams,')
('the', 'ngrams,', 'where')
('ngrams,', 'where', 'n')
('where', 'n', 'is')
('n', 'is', 'the')
('is', 'the', 'length')
('the', 'length', 'of')
('length', 'of', 'the')
('of', 'the', 'sentence.')
('the', 'sentence.', 'In')
('sentence.', 'In', 'short,')
('In', 'short,', 'this')
('short,', 'this', 'function')
('this', 'function', 'generates')
('function', 'generates', 'ngrams')
('generates', 'ngrams', 'for')
('ngrams', 'for', 'all')
('for', 'all', 'possible')
('all', 'possible', 'values')
('possible', 'values', 'of')
('values', 'of', 'n.')
For sentence 1 , trigrams are: 
('Gensim', 'is', 'a')
('is', 'a', 'free')
('a', 'free', 'open-source')
('free', 'open-source', 'Python')
('open-source', 'Python', 'library')
('Python', 'library', 'for')
('library', 'for', 'representing')
('for', 'representing', 'documents')
('representing', 'documents', 'as')
('documents', 'as', 'semantic')
('as', 'semantic', 'vectors,\n')

For sentence 1 , trigrams are: 
('as', 'efficiently', 'and')
('efficiently', 'and', 'painlessly')
('and', 'painlessly', 'as')
('painlessly', 'as', 'possible')

For sentence 2 , trigrams are: 
('', 'Gensim', 'is')
('Gensim', 'is', 'designed')
('is', 'designed', 'to')
('designed', 'to', 'process')
('to', 'process', 'raw,\n')

For sentence 1 , trigrams are: 
('unstructured', 'digital', 'texts')
('digital', 'texts', 'using')
('texts', 'using', 'unsupervised')
('using', 'unsupervised', 'machine')
('unsupervised', 'machine', 'learning')
('machine', 'learning', 'algorithms')

For sentence 2 , trigrams are: 

'''


# Q2. What does n-gram do in Python?
# A. N-grams split the sentence into multiple sequences of tokens depending upon the value of n. For example, given n=3, n-grams for the following sentence “I am doing well today” looks like [“I am doing”, “am doing good”, “doing good today”]

# Q3. What are n-grams used for in NLP?
# A. N-grams are used in the various use cases of NLP, such as spelling correction, machine translation, language models, semantic feature extraction, etc.

# Q4. What is the difference between n-grams and bigrams?
# A. The ‘n’ in n-grams refers to the no. of sequences of tokens. Hence, when the value of n=2, it’s known as bigrams.

# Q5. What are the advantages and disadvantages of using n-grams in NLP?
# A. Here are the advantages and disadvantages of n-grams in NLP.
# Pros
# The concept of n-grams is simple and easy to use yet powerful. Hence, it can be used to build a variety of applications in NLP, like language models, spelling correctors, etc.
# Cons
# N-grams cannot deal Out Of Vocabulary (OOV) words. It works well with the words present in the training set. In the case of an Out Of Vocabulary (OOV) word, n-grams fail to tackle it.
# Another serious concern about n-grams is that it deals with large sparsity.