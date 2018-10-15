import spacy

nlp = spacy.load('en')
doc = nlp(u'Which kitchen items are bowls in the image')
for token in doc:
    print("{0}\t{1}".format(token.text, token.pos_))