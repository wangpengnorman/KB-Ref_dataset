from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
import json
import enchant
from collections import defaultdict
import numpy as np
import time

if __name__ == '__main__':
    d = enchant.Dict("en_US")
    with open('./word2vec_model/Vocabulary.json') as file:
        word_dict = json.load(file)
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize_sentence(sentence, flag=True):
        res = []
        for word, pos in pos_tag(word_tokenize(sentence)):
            if flag:
                res.append(word)
            else:
                wordnet_pos = get_wordnet_pos(pos)
                if wordnet_pos == wordnet.NOUN or wordnet_pos == wordnet.ADJ or wordnet_pos == wordnet.ADV:
                    res.append(word)
        return res


    def sentence(sentence, flag=True):
        a = []
        for e in lemmatize_sentence(sentence, flag=flag):
            if d.check(e):
                e = e.lower()
                if e in word_dict and word_dict[e] < len(word_dict)-1:
                    a.append(word_dict[e])
                else:
                    a.append(len(word_dict)-1)
        if len(a) > 50:
            c = 50
        else:
            c = len(a)
            while len(a) < 50:
                a.append(len(word_dict)-1)
        return a[0:50], c


    with open('./word2vec_model/expression.json') as file:
        data = json.load(file)
    with open('./word2vec_model/candidate.json') as file:
        cand = json.load(file)
    with open('./json/top_facts.json') as file:
        facts = json.load(file)
    with open('./word2vec_model/objects.json') as file:
        objects = json.load(file)
    with open('./word2vec_model/image.json') as file:
        w_h = json.load(file)
    with open('./word2vec_model/train.json') as file:
        train_set = json.load(file)
    with open('./word2vec_model/val.json') as file:
        val_set = json.load(file)
    with open('./word2vec_model/test.json') as file:
        test_set = json.load(file)
    train = []
    val = []
    test = []
    length = []
    print(len(data))
    for k in data:
        try:
            start = time.time()
            label = cand[k.split('_')[0]].index(k.split('_')[1])
            img = k.split('_')[0]
            expression, leng = sentence(data[k][0], flag=True)
            e_mask = leng
            bbox = []
            final_f = []
            length = []
            c_mask = len(cand[k.split('_')[0]]) - cand[k.split('_')[0]].count('-1')
            for c in cand[k.split('_')[0]]:
                if c != '-1':
                    lg = []
                    bbox.append(
                        [objects[img][c][2], objects[img][c][3], objects[img][c][4],
                         objects[img][c][5]])
                    try:
                        fact = facts[k][c]
                        f = []
                        if len(fact) >= 50:
                            for i in range(50):
                                f1, leng = sentence(fact[i])
                                f.append(f1)
                                lg.append(leng)
                        else:
                            for i in range(len(fact)):
                                f1, leng = sentence(fact[i])
                                f.append(f1)
                                lg.append(leng)
                            while len(f) < 50:
                                a = np.ones(50) + len(word_dict)-2
                                f.append(a.tolist())
                                lg.append(0)
                        final_f.append(f)
                        length.append(lg)
                    except:
                        f = []
                        while len(f) < 50:
                            a = np.ones(50) + len(word_dict)-2
                            f.append(a.tolist())
                            lg.append(0)
                        final_f.append(f)
                        length.append(lg)
                else:
                    lg = []
                    bbox.append([0, 0, 0, 0])
                    f = []
                    while len(f) < 50:
                        a = np.ones(50) + len(word_dict)-2
                        f.append(a.tolist())
                        lg.append(0)
                    final_f.append(f)
                    length.append(lg)

            if img in train_set:
                train.append({'image': img,
                              'label': label,
                              'expression': expression,
                              'e_mask': e_mask,
                              'bbox': bbox,
                              'w_h': w_h[k.split('_')[0]],
                              'facts': final_f,
                              'mask': length,
                              'c_mask': c_mask})
            elif img in val_set:
                val.append({'image': img,
                             'label': label,
                             'expression': expression,
                             'e_mask': e_mask,
                             'bbox': bbox,
                             'w_h': w_h[k.split('_')[0]],
                             'facts': final_f,
                             'mask': length,
                             'c_mask': c_mask})
            else:
                test.append({'image': img,
                            'label': label,
                            'expression': expression,
                            'e_mask': e_mask,
                            'bbox': bbox,
                            'w_h': w_h[k.split('_')[0]],
                            'facts': final_f,
                            'mask': length,
                            'c_mask': c_mask})
            print(time.time() - start)
        except:
            continue

    
    with open('./json/train.json', 'w') as file:
        json.dump(train, file)
    with open('./json/val.json', 'w') as file:
        json.dump(val, file)
    with open('./json/test.json', 'w') as file:
        json.dump(test, file)

