# -*- coding: utf-8 -*-
# ============== DIPENDENZE ==============
import sys, codecs, nltk, math
from nltk import FreqDist
from tabulate import tabulate # libreria per output tabulari
# ==============   CLASSI   ==============
class Corpus:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.phrases = None
        self.tokens = self.tokenized_text()
        self.tokens_charSum = sum(len(i) for i in self.tokens)
        self.v_growth = []
        self.h_growth = []
        self.lex_rich = float(len(set(self.tokens[:5000]))) / 5000
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
        with codecs.open(self.path, "r", "utf-8") as corpora:
            raw = corpora.read()
            phrases = sent_tokenizer.tokenize(raw.lower())
            self.phrases = len(phrases)
            c = []
            for p in phrases:
                tokens = nltk.word_tokenize(p)
                c += tokens
        return c

    def pos_distribution(self, request):
        tag_distrib = nltk.FreqDist(tag for (word, tag) in self.pos_tag)
        tot = {'noun': 0, 'adj':0, 'verb':0, 'pron':0}
        for t, n in tag_distrib.most_common(): # tuple unpacking
            if t.startswith('NN'): # sostantivi
                tot['noun'] += n
            elif t.startswith('JJ'):
                tot['adj'] += n
            elif t.startswith('V'):
                tot['verb'] += n
            elif t.startswith(('PR','WH')):
                tot['pron'] += n
        if request is 'percentuali':
            return [ # calcolo percentuali
                tot['noun'] / float(len(self.tokens)) * 100,
                tot['adj'] / float(len(self.tokens)) * 100,
                tot['verb'] / float(len(self.tokens)) * 100,
                tot['pron'] / float(len(self.tokens)) * 100
                ]
        elif request is 'medie':
            return [ # calcolo medie
                tot['noun']/self.phrases,
                tot['adj']/self.phrases,
                tot['verb']/self.phrases,
                tot['pron']/self.phrases
                ]

def main():
    if len(m.tokens) >= len(f.tokens):  # tolgo le ultime 3 cifre del minore
        step = int(len(f.tokens)/1000)
    else:
        step = int(len(m.tokens)/1000)
    m.h_growth, m.v_growth = m.hapax_vocabulary_growth(step)
    f.h_growth, f.v_growth = f.hapax_vocabulary_growth(step)
    # tabella
    print '\n> ANALISI BASILARI <'
    headers = ['\ncorpora', '\nfrasi', '\ntoken', 'media\ntoken/frase', 'media\nchar/token']
    records = [
        [m.name, m.phrases, len(m.tokens), len(m.tokens)/m.phrases, m.tokens_charSum/len(m.tokens)],
        [f.name, f.phrases, len(f.tokens), len(f.tokens)/f.phrases, f.tokens_charSum/len(f.tokens)]
    ]
    print tabulate(records, headers)
    # tabella
    print '\n> CRESCITA VOCABOLARIO OGNI 1000 TOKEN <'
    headers = [m.name, f.name]
    records = zip(m.v_growth, f.v_growth)
    print tabulate(records, headers)
    # tabella
    print '\n> CRESCITA HAPAX OGNI 1000 TOKEN <'
    records = zip(m.h_growth, f.h_growth) # allineo crescita hapax
    print tabulate(records, headers)
    # tabella
    print '\n> TYPE/TOKEN RATIO @ 5000 TOKEN <'
    headers = ['corpora', 'lexical richness']
    records = [ [m.name, m.lex_rich], [f.name, f.lex_rich] ]
    print tabulate(records, headers)
    # tabella
    print '\n> DISTRIBUZIONE % TAG <'
    headers = ['corpora', '%Sostantivi', '%Aggettivi','%Verbi', '%Pronomi']
    records = [ [m.name], [f.name] ]
    for i in range(4):
        records[0].append("%.2f" % m.postag_percent[i])
        records[1].append("%.2f" % f.postag_percent[i])
    print tabulate(records, headers)
    # tabella
    print '\n> MEDIA TAG/FRASE <'
    headers = ['corpora', '%Sostantivi', '%Aggettivi','%Verbi', '%Pronomi']
    records = [[m.name],[f.name]]
    for i in range(4):
        records[0].append(m.medie[i])
        records[1].append(f.medie[i])
    print tabulate(records, headers)

# =====================================================================
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
files = ['TBM.txt', 'TBF.txt']
m = Corpus(files[0], 'travel blog maschi')
f = Corpus(files[1], 'travel blog femmine')
main()