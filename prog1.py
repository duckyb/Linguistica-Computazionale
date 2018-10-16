# -*- coding: utf-8 -*-

import sys
import codecs
import nltk
import math
from nltk import FreqDist
from tabulate import tabulate # libreria per output tabulari

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# >< >< >< >< >< CLASSI >< >< >< >< ><
class Corpus:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.tokens = self.tokenized_text()
        self.phrases = self.phrase_count()
        self.tokens_charSum = self.char_sum()
        self.v_growth = [] # li riempio dopo perché-
        self.h_growth = [] # -non so a quanto mi devo fermare
        self.lex_rich = self.lexical_richness()
        self.pos_tag = nltk.pos_tag(self.tokens)
        self.postag_percent = self.pos_distribution('percentuali')
        self.medie = self.pos_distribution('medie')
        
    def hapax_vocabulary_growth(self, n):
        hg = [0] * n
        vg = [0] * n
        for i in range(n):
            hg[i] = len(FreqDist(self.tokens[:1000 + (i * 1000)]).hapaxes())
            vg[i] = len(set(self.tokens[:1000 + (i * 1000)]))          
        return hg, vg

    def tokenized_text(self): # tokenizza il testo
        corpora = codecs.open(self.path, "r", "utf-8")
        raw = corpora.read()
        phrases = sent_tokenizer.tokenize(raw.lower())
        c = []
        for p in phrases:
            # tokens = nltk.word_tokenize(p.encode('utf8'))
            tokens = nltk.word_tokenize(p)
            c.extend(tokens)
        corpora.close()
        return c

    def char_sum(self): # totale caratteri dei token
        tot = 0
        for t in self.tokens:
            tot += len(t)
        return tot

    def phrase_count(self): # totale frasi nel testo
        corpora = codecs.open(self.path, "r", "utf-8")
        raw = corpora.read()
        phrases = sent_tokenizer.tokenize(raw)
        c = 0
        for p in phrases: # conteggio frasi
            c += 1
        corpora.close()
        return c

    def lexical_richness(self): # ricchezza lessicale @ 5K token
        first_5k = self.tokens[:5000]
        type_token_ratio = float(len(set(first_5k))) / 5000
        return type_token_ratio

    def pos_distribution(self, request):
        tag_distrib = nltk.FreqDist(tag for (word, tag) in self.pos_tag)
        tot = {'noun': 0, 'adj':0, 'verb':0, 'pron':0}
        for t, n in tag_distrib.most_common(): #tuple unpacking
            if t.startswith('NN'): #sostantivi
                tot['noun'] += n
            elif t.startswith('JJ'):
                tot['adj'] += n
            elif t.startswith('V'):
                tot['verb'] += n
            elif t.startswith(('PR','WH')):
                tot['pron'] += n
        if request is 'percentuali':
            percent = {
                'noun': tot['noun'] / float(len(self.tokens)) * 100,
                'adj': tot['adj'] / float(len(self.tokens)) * 100,
                'verb': tot['verb'] / float(len(self.tokens)) * 100,
                'pron': tot['pron'] / float(len(self.tokens)) * 100
            }
            return [
                percent['noun'],
                percent['adj'],
                percent['verb'],
                percent['pron']
                ]
        if request is 'medie':
            return [
                tot['noun']/self.phrases,
                tot['adj']/self.phrases,
                tot['verb']/self.phrases,
                tot['pron']/self.phrases
                ]

def main():
    # istanziazione oggetti
    m = Corpus('TBM.txt', 'travel blog maschi')
    f = Corpus('TBF.txt', 'travel blog femmine')

    if len(m.tokens) >= len(f.tokens):  # tolgo le ultime 3 cifre del minore
        step = int(len(m.tokens)/1000)
    else:
        step = int(len(m.tokens)/1000)
    m.h_growth, m.v_growth = m.hapax_vocabulary_growth(step)
    f.h_growth, f.v_growth = f.hapax_vocabulary_growth(step)
    # tabella
    headers = ['\ncorpora', '\nfrasi', '\ntoken', 'media\ntoken/frase', 'media\nchar/token']
    records = [
        [m.name, m.phrases, len(m.tokens), len(m.tokens)/m.phrases, m.tokens_charSum/len(m.tokens)],
        [f.name, f.phrases, len(f.tokens), len(f.tokens)/f.phrases, f.tokens_charSum/len(f.tokens)]
    ]
    print tabulate(records, headers, tablefmt = 'psql')
    # tabella
    print '\n> CRESCITA VOCABOLARIO OGNI 1000 TOKEN <'
    headers = [m.name, f.name]
    records = zip(m.v_growth, f.v_growth)
    print tabulate(records, headers, tablefmt = 'psql')
    # tabella
    print '\n> CRESCITA HAPAX OGNI 1000 TOKEN <'
    records = zip(m.h_growth, f.h_growth) # allineo crescita hapax
    print tabulate(records, headers, tablefmt = 'psql')
    # tabella
    print '\n> TYPE/TOKEN RATIO @ 5000 TOKEN <'
    headers = ['corpora', 'lexical richness']
    records = [
        [m.name, m.lex_rich],
        [f.name, f.lex_rich]
    ]
    print tabulate(records, headers, tablefmt = 'psql')
    # tabella
    print '\n> DISTRIBUZIONE % TAG <'
    headers = ['corpora', '%Sostantivi', '%Aggettivi','%Verbi', '%Pronomi']
    records = [[m.name],[f.name]]
    for i in range(4):
        records[0].append("%.2f" % m.postag_percent[i])
        records[1].append("%.2f" % f.postag_percent[i])
    print tabulate(records, headers, tablefmt = 'psql')
    # tabella
    print '\n> MEDIA TAG/FRASE <'
    headers = ['corpora', '%Sostantivi', '%Aggettivi','%Verbi', '%Pronomi']
    records = [[m.name],[f.name]]
    for i in range(4):
        records[0].append(m.medie[i])
        records[1].append(f.medie[i])
    print tabulate(records, headers, tablefmt = 'psql')

main()