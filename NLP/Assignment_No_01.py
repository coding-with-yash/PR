'''
Assignment No: 01
Name: Yash Pramod Dhage
Roll: 70
Batch: B4
Title: Text Pre-Processing Using NLP Operation:
        Perform Tokenization,Lemmatization,Stop Word,Part-of-speech tagging use any sample text.
'''

import spacy                        # import Spacy library
nlp = spacy.load("en_core_web_sm")  # Load english Dictionary

# 01 Token
print("Tokenization")
about_text = (
    "Yash Dhage is an aspiring data scientist"
    " living in Nashik City."
    " He loves exploring machine learning"
    " and is passionate about data analysis."
)
about_doc = nlp(about_text)
for token in about_doc:
    print (token, token.idx)


#  02 Stop Word
print("Stop Word Removal ")
about_doc = nlp(about_text)
print([token for token in about_doc if not token.is_stop])


# 03 Lemmatization
print("Lemmatization")
conference_help_doc = nlp(about_text)
for token in conference_help_doc:
    if str(token) != str(token.lemma_):
        print(f"{str(token):>20} : {str(token.lemma_)}")


# 04 Part-of-speech tagging (POS tag)
print("Part-of-speech tagging")
about_doc = nlp(about_text)
for token in about_doc:
    print(
        f"""
TOKEN: {str(token)}
=====
TAG: {str(token.tag_):10} POS: {token.pos_}
EXPLANATION: {spacy.explain(token.tag_)}"""
    )




#Output:
# Tokenization
#
# Yash 0
# Dhage 5
# is 11
# an 14
# aspiring 17
# data 26
# scientist 31
# living 41
# in 48
# Nashik 51
# City 58
# . 62
# He 64
# loves 67
# exploring 73
# machine 83
# learning 91
# and 100
# is 104
# passionate 107
# about 118
# data 124
# analysis 129
# . 137
#
# Stop Word
# [Yash, Dhage, aspiring, data, scientist, living, Nashik, City, ., loves, exploring, machine, learning, passionate, data, analysis, .]
#
# Lemmatization
#                   is : be
#             aspiring : aspire
#               living : live
#                   He : he
#                loves : love
#            exploring : explore
#                   is : be
# Part-of-speech tagging
#
# TOKEN: Yash
# =====
# TAG: NNP        POS: PROPN
# EXPLANATION: noun, proper singular
#
# TOKEN: Dhage
# =====
# TAG: NNP        POS: PROPN
# EXPLANATION: noun, proper singular
#
# TOKEN: is
# =====
# TAG: VBZ        POS: AUX
# EXPLANATION: verb, 3rd person singular present
#
# TOKEN: an
# =====
# TAG: DT         POS: DET
# EXPLANATION: determiner
#
# TOKEN: aspiring
# =====
# TAG: VBG        POS: VERB
# EXPLANATION: verb, gerund or present participle
#
# TOKEN: data
# =====
# TAG: NN         POS: NOUN
# EXPLANATION: noun, singular or mass
#
# TOKEN: scientist
# =====
# TAG: NN         POS: NOUN
# EXPLANATION: noun, singular or mass
#
# TOKEN: living
# =====
# TAG: VBG        POS: VERB
# EXPLANATION: verb, gerund or present participle
#
# TOKEN: in
# =====
# TAG: IN         POS: ADP
# EXPLANATION: conjunction, subordinating or preposition
#
# TOKEN: Nashik
# =====
# TAG: NNP        POS: PROPN
# EXPLANATION: noun, proper singular
#
# TOKEN: City
# =====
# TAG: NNP        POS: PROPN
# EXPLANATION: noun, proper singular
#
# TOKEN: .
# =====
# TAG: .          POS: PUNCT
# EXPLANATION: punctuation mark, sentence closer
#
# TOKEN: He
# =====
# TAG: PRP        POS: PRON
# EXPLANATION: pronoun, personal
#
# TOKEN: loves
# =====
# TAG: VBZ        POS: VERB
# EXPLANATION: verb, 3rd person singular present
#
# TOKEN: exploring
# =====
# TAG: VBG        POS: VERB
# EXPLANATION: verb, gerund or present participle
#
# TOKEN: machine
# =====
# TAG: NN         POS: NOUN
# EXPLANATION: noun, singular or mass
#
# TOKEN: learning
# =====
# TAG: NN         POS: NOUN
# EXPLANATION: noun, singular or mass
#
# TOKEN: and
# =====
# TAG: CC         POS: CCONJ
# EXPLANATION: conjunction, coordinating
#
# TOKEN: is
# =====
# TAG: VBZ        POS: AUX
# EXPLANATION: verb, 3rd person singular present
#
# TOKEN: passionate
# =====
# TAG: JJ         POS: ADJ
# EXPLANATION: adjective (English), other noun-modifier (Chinese)
#
# TOKEN: about
# =====
# TAG: IN         POS: ADP
# EXPLANATION: conjunction, subordinating or preposition
#
# TOKEN: data
# =====
# TAG: NN         POS: NOUN
# EXPLANATION: noun, singular or mass
#
# TOKEN: analysis
# =====
# TAG: NN         POS: NOUN
# EXPLANATION: noun, singular or mass
#
# TOKEN: .
# =====
# TAG: .          POS: PUNCT
# EXPLANATION: punctuation mark, sentence closer



# 1.  python -m venv venv
# 2.  .\venv\Scripts\activate
# 3.  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# 4.  pip install spacy
# 5 . python -m spacy download en_core_web_sm




# Text pre-processing is a crucial step in natural language processing (NLP) that involves cleaning and 
# transforming raw text data into a format that is suitable for analysis. Several techniques are commonly used in text pre-processing,
# including tokenization, lemmatization, stop word removal, and part-of-speech tagging. Let's explore each of these concepts:

# 1. **Tokenization:**
#    - **Definition:** Tokenization is the process of breaking down a text into smaller units, typically words or phrases, known as tokens.
#    - **Example:** The sentence "ChatGPT is amazing!" would be tokenized into the following tokens: ["ChatGPT", "is", "amazing", "!"].

# 2. **Lemmatization:**
#    - **Definition:** Lemmatization is the process of reducing words to their base or root form, called the lemma, to normalize variations.
#    - **Example:** The lemma of the words "running," "ran," and "runs" is "run."

# 3. **Stop Words:**
#    - **Definition:** Stop words are common words that are often removed from text data during pre-processing because they are considered 
#                       to carry little meaningful information and are used frequently in a language.
#    - **Example:** Common stop words in English include "the," "and," "is," "in," etc.

# 4. **Part-of-Speech (POS) Tagging:**
#    - **Definition:** Part-of-speech tagging involves assigning a grammatical category (such as noun, verb, adjective, etc.) to each word in a sentence.
#    - **Example:** In the sentence "The cat is sleeping," part-of-speech tagging would involve labeling "The" as a determiner, "cat" as a noun, "is" as a verb, and "sleeping" as an adjective.

# Text pre-processing is often performed in a sequential manner, where tokenization is typically the first step, followed by lemmatization, stop word removal, and part-of-speech tagging, depending on the specific requirements of the NLP task at hand. These techniques help in reducing the dimensionality of the data, removing noise, and extracting relevant features for downstream NLP tasks such as text classification, sentiment analysis, and information retrieval.