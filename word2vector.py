import json
from gensim.models import word2vec
from nltk import word_tokenize, sent_tokenize
import enchant
import time


if __name__ == '__main__':
    with open('./data/KB-REF/Wikipedia.json') as file:
        Wikipedia = json.load(file)
    with open('./data/KB-REF/ConceptNet.json') as file:
        ConceptNet = json.load(file)
    with open('./data/KB-REF/WebChild.json') as file:
        WebChild = json.load(file)
    out = ''
    l = 0
    for k in Wikipedia:
        fact = sent_tokenize(Wikipedia[k])
        l += len(fact)
        for f in fact:
            out += f
            out += '\n'
    for k in ConceptNet:
        fact = sent_tokenize(ConceptNet[k].replace('.', '. ').replace('has/have ', ''))
        l += len(fact)
        for f in fact:
            out += f
            out += '\n'
    for k in WebChild:
        fact = sent_tokenize(WebChild[k])
        l += len(fact)
        for f in fact:
            out += f
            out += '\n'
    with open('./data/KB-REF/f.txt', 'w', encoding='utf-8') as file:
        file.write(out)
    file.close()
    
    start = time.time()
    sentences = word2vec.Text8Corpus('./data/KB-REF/f.txt')
    model = word2vec.Word2Vec(sentences, size=300, hs=1, sg=1, min_count=5, window=5, iter=100, workers=4)
    model.save('./word2vec_model/facts.model')
    print(time.time()-start)
