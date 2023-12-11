# Title : Named Entity Recognition (NER) using spacy.
import spacy
raw_text=""" """
NER = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
text= NER(raw_text)
for w in text.ents:
    print(w.text,w.label_)
    
spacy.displacy.render(text, style="ent",jupyter=True)
spacy.explain(u"NORP")
#Detecting the entities from the text
#Classifying them into different categories

#=================================================================================
# TITLE: Generating Unigrams,Bigrams and Trigrams in nltk.
from nltk.util import ngrams
def n_grams(n):
    #sample input
    para=''' . '''
    grams=ngrams(para.split(),n)

    for i in grams:
        print(i)
print("Unigrams:")
n_grams(1)
print("\nBigrams:")
n_grams(2)
print("\nTrigrams:")
n_grams(3)

#===============================================================
from nltk import ngrams


def n_g(n):
  file = open("sample.txt")
  for i in file.readlines():
    cumulative = i
    sentences = i.split(".")
    counter = 0
    for sentence in sentences:
      print("For sentence", counter + 1, n, "grams are: ")
      trigrams = ngrams(sentence.split(" "), n)
      for grams in trigrams:
        print(grams)
      counter += 1
      print()


print("Unigrams:")
n_g(1)
print("\nBigrams:")
n_g(2)
print("\nTrigrams:")
n_g(3)

Language modeling is the way of determining the probability of any sequence of words.
probabilistic models that are able to predict the next word
