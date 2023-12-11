'''
Assignment No: 03 (NLTK)
Name: Yash Pramod Dhage
Roll: 70
Batch: B4
Title: Implements Named Entity Recognition(NER) on textual data using SpaCy or NLTK library.
'''

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.chunk import tree2conlltags

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


raw_text=("The Board of Control for Cricket in India (BCCI) is the governing body for cricket in India"
          " and is under the jurisdiction of Ministry of Youth Affairs and Sports, Government of India. "
          "[2] The board was formed in December 1928 as a society, registered under the Tamil Nadu Societies "
          "Registration Act. It is a consortium of state cricket associations and the state associations select "
          "their representatives who in turn elect the BCCI Chief. Its headquarters are in Wankhede Stadium, Mumbai."
          "Grant Govan was its first president and Anthony De Mello its first secretary.")


raw_words = word_tokenize(raw_text)         # Tokenize the input text

tags = pos_tag(raw_words)                   # Perform part-of-speech tagging on the words

ne = ne_chunk(tags, binary=True)            # Apply named entity recognition

iob = tree2conlltags(ne)                    # Convert the named entity chunks to IOB format

for word, pos, tag in iob:                  # Print the IOB tags
    print(word, pos, tag)


'''
OUTPUT

The DT O
Board NNP B-NE
of IN O
Control NNP B-NE
for IN O
Cricket NNP O
in IN O
India NNP B-NE
( ( O
BCCI NNP B-NE
) ) O
is VBZ O
the DT O
governing VBG O
body NN O
for IN O
cricket NN O
in IN O
India NNP B-NE
and CC O
is VBZ O
under IN O
the DT O
jurisdiction NN O
of IN O
Ministry NNP B-NE
of IN O
Youth NNP B-NE
Affairs NNPS I-NE
and CC O
Sports NNP B-NE
, , O
Government NNP O
of IN O
India NNP B-NE
. . O
[ CC O
2 CD O
] VBP O
The DT O
board NN O
was VBD O
formed VBN O
in IN O
December NNP O
1928 CD O
as IN O
a DT O
society NN O
, , O
registered VBN O
under IN O
the DT O
Tamil NNP B-NE
Nadu NNP I-NE
Societies NNP I-NE
Registration NNP O
Act NNP O
. . O
It PRP O
is VBZ O
a DT O
consortium NN O
of IN O
state NN O
cricket NN O
associations NNS O
and CC O
the DT O
state NN O
associations NNS O
select VBP O
their PRP$ O
representatives NNS O
who WP O
in IN O
turn NN O
elect VBP O
the DT O
BCCI NNP B-NE
Chief NNP I-NE
. . O
Its PRP$ O
headquarters NNS O
are VBP O
in IN O
Wankhede NNP B-NE
Stadium NNP I-NE
, , O
Mumbai.Grant NNP O
Govan NNP O
was VBD O
its PRP$ O
first JJ O
president NN O
and CC O
Anthony NNP B-NE
De NNP I-NE
Mello NNP I-NE
its PRP$ O
first JJ O
secretary NN O
. . O

'''