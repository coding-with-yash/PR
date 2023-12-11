"""
    Assignment No: 06
    Name: Yash Pramod Dhage
    Roll: 70
    Batch: B4
    Practical : Implement and visualize Dependency Parsing of Textual Input using Stan- ford CoreNLP and Spacy library
"""
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
piano_text = "My school starts at 8:00.We always eat dinner together.They take the bus to work. He doesn't like vegetables."
piano_doc = nlp(piano_text)
for token in piano_doc:
    print( f"""
TOKEN: {token.text}
{token.tag_ = }
{token.head.text = }
{token.dep_ = }"""
    )
displacy.serve(piano_doc, style="dep")
#OUTPUT:
"""
TOKEN: My
token.tag_ = 'PRP$'
token.head.text = 'school'
token.dep_ = 'poss'

TOKEN: school
token.tag_ = 'NN'
token.head.text = 'starts'
token.dep_ = 'nsubj'

TOKEN: starts
token.tag_ = 'VBZ'
token.head.text = 'eat'
token.dep_ = 'nsubj'

TOKEN: at
token.tag_ = 'IN'
token.head.text = 'starts'
token.dep_ = 'prep'

TOKEN: 8:00.We
token.tag_ = 'NNS'
token.head.text = 'at'
token.dep_ = 'pobj'

TOKEN: always
token.tag_ = 'RB'
token.head.text = 'eat'
token.dep_ = 'advmod'

TOKEN: eat
token.tag_ = 'VBP'
token.head.text = 'eat'
token.dep_ = 'ROOT'

TOKEN: dinner
token.tag_ = 'NN'
token.head.text = 'eat'
token.dep_ = 'dobj'

TOKEN: together
token.tag_ = 'RB'
token.head.text = 'eat'
token.dep_ = 'advmod'

TOKEN: .
token.tag_ = '.'
token.head.text = 'eat'
token.dep_ = 'punct'

TOKEN: They
token.tag_ = 'PRP'
token.head.text = 'take'
token.dep_ = 'nsubj'

TOKEN: take
token.tag_ = 'VBP'
token.head.text = 'take'
token.dep_ = 'ROOT'

TOKEN: the
token.tag_ = 'DT'
token.head.text = 'bus'
token.dep_ = 'det'

TOKEN: bus
token.tag_ = 'NN'
token.head.text = 'take'
token.dep_ = 'dobj'

TOKEN: to
token.tag_ = 'TO'
token.head.text = 'work'
token.dep_ = 'aux'

TOKEN: work
token.tag_ = 'VB'
token.head.text = 'take'
token.dep_ = 'xcomp'

TOKEN: .
token.tag_ = '.'
token.head.text = 'take'
token.dep_ = 'punct'

TOKEN: He
token.tag_ = 'PRP'
token.head.text = 'like'
token.dep_ = 'nsubj'

TOKEN: does
token.tag_ = 'VBZ'
token.head.text = 'like'
token.dep_ = 'aux'

TOKEN: n't
token.tag_ = 'RB'
token.head.text = 'like'
token.dep_ = 'neg'

TOKEN: like
token.tag_ = 'VB'
token.head.text = 'like'
token.dep_ = 'ROOT'

TOKEN: vegetables
token.tag_ = 'NNS'
token.head.text = 'like'
token.dep_ = 'dobj'

TOKEN: .
token.tag_ = '.'
token.head.text = 'like'
token.dep_ = 'punct'
"""




# Q1. What is dependency parsing?
# A. Dependency parsing is a linguistic analysis technique used in natural language processing to uncover grammatical relationships between words in a sentence. It involves parsing a sentence’s structure to create a tree-like representation that shows how words depend on one another. This helps reveal the syntactic structure, roles of words (like subjects and objects), and overall meaning within the sentence.

# Q2. What is dependency parsing and syntactic parsing?
# A. Dependency parsing and syntactic parsing are linguistic analysis methods used in natural language processing. Dependency parsing focuses on revealing grammatical relationships between words in a sentence, portraying how words depend on each other. It constructs a tree structure that illustrates these dependencies, aiding in understanding sentence structure. Syntactic parsing, broader in scope, aims to uncover the overall syntactic structure of a sentence, encompassing phrase boundaries, constituents, and grammatical rules. Both techniques play a crucial role in extracting meaning and insights from text data, benefiting various language processing tasks.

# Conclusion
# Organizations are seeking new methods to make use of computer technology as it advances beyond its artificial limits. A significant rise in computing speeds and capacities has resulted in the development of new and highly intelligent software systems, some of which are ready to replace or enhance human services.

# One of the finest examples is the growth of natural language processing (NLP), with smart chatbots prepared to change the world of customer service and beyond.

# In summary, human language is awe-inspiringly complex and diverse.

# In addition to assisting in the resolution of linguistic ambiguity, NLP is significant because it offers a helpful mathematical foundation for a variety of downstream applications such as voice recognition and text analytics.

# In order to understand NLP, it’s important to have a good understanding of the basics, Dependency Parsing is one of them.

