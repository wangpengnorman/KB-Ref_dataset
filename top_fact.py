from gensim.models import word2vec
import json
from collections import defaultdict
from nltk import sent_tokenize, word_tokenize
import numpy as np
import random
import time


if __name__ == '__main__':
    model = word2vec.Word2Vec.load('./word2vec_model/facts.model')
    with open('./data/KB-REF/expression.json') as file:
        expression = json.load(file)
    with open('./data/KB-REF/candidate.json') as file:
        cand = json.load(file)
    with open('./data/KB-REF/Wikipedia.json') as file:
        Wikipedia = json.load(file)
    with open('./data/KB-REF/ConceptNet.json') as file:
        ConceptNet = json.load(file)
    with open('./data/KB-REF/WebChild.json') as file:
        WebChild = json.load(file)
    with open('./data/KB-REF/objects.json') as file:
        objects = json.load(file)

    top_facts = {}
    for k in expression:
        start = time.time()
        middle = {}
        img = k.split('_')[0]
        e = expression[k][0]
        candidates = cand[img]
        j = 0
        em = np.zeros(300)
        for f in word_tokenize(e):
            try:
                em += np.array(model.wv.get_vector(f.lower()))
                j += 1
            except:
                continue
        em /= j
        for c in candidates:
            if c != '-1':
                sims = []
                fs = []
                final = []
                o = objects[img][c][0].split('.')[0]
                try:
                    facts = sent_tokenize(Wikipedia[o.lower()])
                    for fact in facts:
                        j = 0
                        nm = np.zeros(300)
                        for f in word_tokenize(fact):
                            try:
                                nm += np.array(model.wv.get_vector(f.lower()))
                                j += 1
                            except:
                                continue
                        if j!= 0:
                            nm /= j
                            sim = np.dot(em, nm) / (np.linalg.norm(em) * np.linalg.norm(nm))
                            sim = 0.5 + 0.5 * sim
                            fs.append(fact)
                            sims.append(sim)
                except:
                    continue
                try:
                    facts = sent_tokenize(ConceptNet[o.lower()].replace('.', '. ').replace('has/have ', ''))
                    for fact in facts:
                        j = 0
                        nm = np.zeros(300)
                        for f in word_tokenize(fact):
                            try:
                                nm += np.array(model.wv.get_vector(f.lower()))
                                j += 1
                            except:
                                continue
                        if j!= 0:
                            nm /= j
                            sim = np.dot(em, nm) / (np.linalg.norm(em) * np.linalg.norm(nm))
                            sim = 0.5 + 0.5 * sim
                            fs.append(fact)
                            sims.append(sim)
                except:
                    continue
                try:
                    facts = sent_tokenize(WebChild[o.lower()])
                    for fact in facts:
                        j = 0
                        nm = np.zeros(300)
                        for f in word_tokenize(fact):
                            try:
                                nm += np.array(model.wv.get_vector(f.lower()))
                                j += 1
                            except:
                                continue
                        if j!= 0:
                            nm /= j
                            sim = np.dot(em, nm) / (np.linalg.norm(em) * np.linalg.norm(nm))
                            sim = 0.5 + 0.5 * sim
                            fs.append(fact)
                            sims.append(sim)
                except:
                    continue
                sims = np.array(sims)
                inxs = np.argsort(-sims)[0:50]
                for ix in inxs:
                    final.append(fs[ix])
                random.shuffle(final)
                middle = dict(middle, **{c: final})

        top_facts = dict(top_facts, **{k: middle})
        print(time.time()-start)

    with open('./json/top_facts.json', 'w') as file:
        json.dump(top_facts, file)

