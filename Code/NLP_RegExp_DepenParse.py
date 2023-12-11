# Regular Expression for URL's,ip addresses,PAN number and Dates.
import re
def find_entities(text):
    result = {
        'URLs': re.findall(r'https?://\S+|www\.\S+', text),
        'IP Addresses': re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text),
        'Dates': re.findall(r'([1-9]|[12][0-9]|3[01])\/(0[1-9]|1[1,2])\/(19|20)\d{2}', text),
        'PAN Numbers': re.findall(r'[A-Z]{5}[0-9]{4}[A-Z]', text),
    }
    return result
text = """
First Dataset:
Visit our website at https://www.google.com.
For support, contact us at support@example.com.
IP address: 192.168.0.2
Date: 27/11/2023
PAN number: GIWPM3635J """

r = find_entities(text)
for entity_type, entities in r.items():
    print(f"{entity_type}: {entities}")
#Regular expressions or RegEx is a sequence of characters mainly used to find or replace patterns embedded in the text.

#Title: Implement and visualize dependency parsing of textual input using stanford coreNLP and spaCy library.
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
texts = ["Mary plays the violin"]
for text in texts:
    doc = nlp(text)
    print(f"Dependency parse visualization for text: '{text}'")
    for token in doc:
        print(f'''
        TOKEN: {token.text}
                =====
                {token.tag_ = }
                {token.head.text = }
                {token.dep_ = }''')      
displacy.serve(doc, style="dep")
#Dependency parsing is a linguistic analysis technique used in natural language processing to uncover grammatical relationships between words in a sentence. It involves parsing a sentence’s structure to create a tree-like representation that shows how words depend on one another. 
This helps reveal the syntactic structure, roles of words (like subSyntactic parsing, broader in scope, aims to uncover the overall syntactic structure of a sentence, encompassing phrase boundaries, constituents, and grammatical rules.
                                                           