'''
Assignment No: 03 (Spacy)
Name: Yash Pramod Dhage
Roll: 70
Batch: B4
Title: Implements Named Entity Recognition(NER) on textual data using SpaCy or NLTK library.
'''



import spacy

raw_text=("The Board of Control for Cricket in India (BCCI) is the governing body for cricket in India"
          " and is under the jurisdiction of Ministry of Youth Affairs and Sports, Government of India. "
          "[2] The board was formed in December 1928 as a society, registered under the Tamil Nadu Societies "
          "Registration Act. It is a consortium of state cricket associations and the state associations select "
          "their representatives who in turn elect the BCCI Chief. Its headquarters are in Wankhede Stadium, Mumbai."
          "Grant Govan was its first president and Anthony De Mello its first secretary.")

NER = spacy.load("en_core_web_sm",disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

text = NER(raw_text)

for x in text.ents:
    print(x.text,x.label_)

spacy.displacy.render(text, style="ent", jupyter=True)

spacy.explain(u"NORP")


'''
Output

The Board of Control for Cricket ORG
India GPE
BCCI ORG
India GPE
Ministry of Youth Affairs ORG
Sports, Government of India ORG
2 CARDINAL
December 1928 DATE
the Tamil Nadu Societies Registration Act ORG
BCCI ORG
Wankhede Stadium FAC
Mumbai GPE
Grant Govan PERSON
first ORDINAL
Anthony De Mello PERSON
first ORDINAL
'''


# !python -m spacy download en_core_web_sm


# Q1. What is named entity recognition with an example?
# Ans. Named Entity Recognition (NER) is an NLP technique that identifies and classifies named entities in text, like names of people, places, organizations, dates, monetary values, etc. For example, in “Apple Inc. was founded by Steve Jobs in Cupertino,” NER would identify “Apple Inc.” as an organization and “Steve Jobs” as a person.

# Q2. What is the purpose of named entity recognition?
# Ans. The purpose of Named Entity Recognition (NER) is to identify and classify named entities within text. It helps extract and understand important information from text data for purposes such as information retrieval, document summarization, question answering, sentiment analysis, machine translation, language modeling, etc.

# Q3. What are the techniques of NER?
# Ans. Some commonly used techniques of NER include rule-based NER, statistical NER, machine learning-based NER, deep learning-based NER, BERT-based models, and hybrid approaches.

# Q4. What are the different types of NER?
# Ans. The four main types of NER are dictionary-based, rule-based, machine learning-based, and deep learning-based.