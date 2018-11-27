# -*- coding: utf-8 -*-
# ============== DIPENDENZE ==============
import sys, codecs, nltk, math
from nltk import FreqDist, word_tokenize, pos_tag
from tabulate import tabulate # libreria per output tabulari
# ==============   CLASSI   ==============
class Corpus:
    def __init__(self, path, name):
        self.path = path # percorso del corpus
        self.name = name # nome del corpus
        self.phrases = None # suddivisione in frasi
        self.tokens = self.tokenized_text() # suddivisione in tokens
        self.tokens_charSum = sum(len(i) for i in self.tokens) # totale di caratteri dei token
        self.v_growth = [] # vocabulary growth
        self.h_growth = [] # hapax growth
        self.lex_rich = float(len(set(self.tokens[:5000]))) / 5000 # lexical richness
        self.pos_tags = pos_tag(self.tokens) # suddivisione in part-of-speech tags
        self.postag_percent = self.pos_distribution('percentuali') # percentuali di part-of-speech tags
        self.medie = self.pos_distribution('medie') # media x frase di part-of-speech

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
            return [token for p in phrases for token in word_tokenize(p)] # comprensione di lista annidata

    def pos_distribution(self, request):
        tag_distrib = FreqDist(tag for (word, tag) in self.pos_tags)
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
            return [tot[key]/float(len(self.tokens))*100 for key in tot]
        elif request is 'medie':
            return [tot[key]/float(self.phrases) for key in tot]

def main():
# ========================= TABELLE =========================
    if len(m.tokens) >= len(f.tokens):  # tolgo le ultime 3 cifre del minore
        step = int(len(f.tokens)/1000)
    else:
        step = int(len(m.tokens)/1000)
    m.h_growth, m.v_growth = m.hapax_vocabulary_growth(step)
    f.h_growth, f.v_growth = f.hapax_vocabulary_growth(step)
    # tabella analisi più semplici
    print '\n> ANALISI BASILARI <'
    headers = ['\ncorpora', '\nfrasi', '\ntoken', 'media\ntoken/frase', 'media\nchar/token']
    records = [
        [m.name, m.phrases, len(m.tokens), len(m.tokens)/m.phrases, m.tokens_charSum/len(m.tokens)],
        [f.name, f.phrases, len(f.tokens), len(f.tokens)/f.phrases, f.tokens_charSum/len(f.tokens)]
    ]
    print tabulate(records, headers)
    # tabella crescita vocabolario
    print '\n> CRESCITA VOCABOLARIO OGNI 1000 TOKEN <'
    headers = [m.name, f.name]
    records = zip(m.v_growth, f.v_growth)
    print tabulate(records, headers)
    # tabella crescita hapax
    print '\n> CRESCITA HAPAX OGNI 1000 TOKEN <'
    records = zip(m.h_growth, f.h_growth) # allineo crescita hapax
    print tabulate(records, headers)
    # tabella (type : token) ratio
    print '\n> TYPE/TOKEN RATIO @ 5000 TOKEN <'
    headers = ['corpora', 'lexical richness']
    records = [ [m.name, m.lex_rich], [f.name, f.lex_rich] ]
    print tabulate(records, headers)
    # tabella distribuzione di part-of-speech tags
    print '\n> DISTRIBUZIONE % TAG <'
    headers = ['corpora', '%Sostantivi', '%Aggettivi','%Verbi', '%Pronomi']
    records = [ [m.name], [f.name] ]
    for i in range(4):
        records[0].append("%.2f" % m.postag_percent[i])
        records[1].append("%.2f" % f.postag_percent[i])
    print tabulate(records, headers)
    # tabella media di part-of-speech tags per frase
    print '\n> MEDIA TAG/FRASE <'
    headers = ['corpora', 'Sostantivi', 'Aggettivi','Verbi', 'Pronomi']
    records = [[m.name],[f.name]]
    for i in range(4):
        records[0].append(m.medie[i])
        records[1].append(f.medie[i])
    print tabulate(records, headers, floatfmt=".2f")
    
# ========================= GLOBAL SCOPE =========================
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')  # scelgo il tokenizzatore
files = ['TBM.txt', 'TBF.txt']                                      # scelgo i due file da confrontare
m = Corpus(files[0], 'travel blog maschi')                          # istanzio il primo corpus
f = Corpus(files[1], 'travel blog femmine')                         # istanzio il secondo
main()                                                              # avvio del programma
