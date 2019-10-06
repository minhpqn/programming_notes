"""
Visualize dependency trees returned by Ginza
"""
import spacy
from spacy import displacy

nlp = spacy.load('ja_ginza')

while True:
    sen = input('Enter the sentence: ')
    sen = sen.strip()
    doc = nlp(sen)
    displacy.serve(doc, style="dep")


