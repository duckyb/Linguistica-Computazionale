# -*- coding: utf-8 -*-

import sys
import codecs
import nltk
import math
from nltk import FreqDist
from tabulate import tabulate # libreria per output tabulari

def tokenized_text(f):
    corpora = codecs.open(f, "r", "utf-8")
    raw = corpora.read()
    phrases = sent_tokenizer.tokenize(raw)
    c = []
    for p in phrases:
        tokens = nltk.word_tokenize(p.encode('utf8'))
        c.extend(tokens)
    corpora.close()
    return c
        
def phrase_count(f):
    corpora = codecs.open(f, "r", "utf-8")
    raw = corpora.read()
    phrases = sent_tokenizer.tokenize(raw)
    count = 0
    for p in phrases: # conteggio frasi
        count += 1
    corpora.close()
    return count

def token_count(f):
    if f == files[0]:
        return len(tokenized['m'])
    if f == files[1]:
        return len(tokenized['f'])

def find_vocabulary(f):
    if f == files[0]:
        return len(set(tokenized['m']))
    if f == files[1]:
        return len(set(tokenized['f']))

def total_token_chars(f):
    if f == files[0]:
        char_sum = 0
        for t in tokenized['m']:
            char_sum += len(t)
        return char_sum
    if f == files[1]:
        char_sum = 0
        for t in tokenized['f']:
            char_sum += len(t)
        return char_sum

def hapax_growth(f):
    if f == files[0]:
        vocabulary = tokenized['m']
    elif f == files[1]:
        vocabulary = tokenized['f']
    h_growth = [0, 0, 0, 0, 0]
    for i in range(len(h_growth)):
        h_growth[i] = len(FreqDist(vocabulary[:1000 + (i * 1000)]).hapaxes())
    return h_growth

def lexical_richness(f):
    if f == files[0]:
        first5k = tokenized['m'][:5000]
        type_token_ratio = float(len(set(first5k))) / 5000
    elif f == files[1]:
        first5k = tokenized['f'][:5000]
        type_token_ratio = float(len(set(first5k))) / 5000
    return type_token_ratio

def pos_tag(f, request):
    if f == files[0]:
        tagged_tokens = nltk.pos_tag(tokenized['m'])
        tokenized_n = tokenized['m']
    elif f == files[1]:
        tagged_tokens = nltk.pos_tag(tokenized['f'])
        tokenized_n = tokenized['f']
    tag_distr = nltk.FreqDist(tag for (word, tag) in tagged_tokens)
    # noun percentage
    total_nouns = 0
    total_adjectives = 0
    total_verbs = 0
    total_pron = 0
    for t, n in tag_distr.most_common(): # tuple unpacking
        if t.startswith('NN'): # sostantivi
            total_nouns += n
        elif t.startswith('JJ'): # aggettivi
            total_adjectives += n
        elif t.startswith('V'): # verbi
            total_verbs += n
        elif t.startswith(('PR', 'WH')): # pronomi
            total_pron += n
    noun_percent = total_nouns / float(len(tokenized_n)) * 100
    adjective_percent = total_adjectives / float(len(tokenized_n)) * 100
    verb_percent = total_verbs / float(len(tokenized_n)) * 100
    pron_percent = total_pron / float(len(tokenized_n)) * 100
    if request == 'percentuali':
        return [noun_percent, adjective_percent, verb_percent, pron_percent]
    if request == 'medie':
        tot = phrase_count(f)
        return [total_nouns/tot, total_adjectives/tot, total_verbs/tot, total_pron/tot]



def main():
    # TABELLA OUTPUT 1
    headers = [ # intestazione tabella
        'Corpora',
        '#Frasi',
        '#Token',
        '#Token/Frase',
        '#Caratteri/Token',
        '|V|'
        ]

    travelblog_m_table = [ # primo record
        'Travel Blog Maschi',
        phrase_count(files[0]), # totale frasi
        token_count(files[0]), # totale token
        token_count(files[0]) / phrase_count(files[0]), # media token / frase
        total_token_chars(files[0]) / token_count(files[0]), # media caratteri / parola
        find_vocabulary(files[0]) # vocabolario
        ]

    travelblog_f_table = [ # secondo record
        'Travel Blog Femmine',
        phrase_count(files[1]), # totale frasi
        token_count(files[1]), # totale token
        token_count(files[1]) / phrase_count(files[1]), # media token / frase
        total_token_chars(files[1]) / token_count(files[1]), # media caratteri / parola
        find_vocabulary(files[1]) # vocabolario
        ]
    # stampa tabella
    print tabulate([travelblog_m_table, travelblog_f_table], headers) #tablefmt='orgtbl'

    # TABELLA OUTPUT 2
    print '\n'

    headers2 = [ # intestazione tabella
        'Corpora',
        'Hapax @1000',
        'Hapax @2000',
        'Hapax @3000',
        'Hapax @4000',
        'Hapax @5000'
        ]

    travel_m_growth = [ # primo record
        'Travel Blog Maschi'
        ]
    travel_m_growth.extend(hapax_growth(files[0]))

    travel_f_growth = [ # secondo record
        'Travel Blog Femmine'
        ]
    travel_f_growth.extend(hapax_growth(files[1]))
    # stampa tabella
    print tabulate([travel_m_growth, travel_f_growth], headers2)

    # TABELLA OUTPUT 3
    print '\n'

    headers3 = [ # intestazione tabella
        'Corpora',
        'Lex_Richness',
        '%Sostantivi',
        '%Aggettivi',
        '%Verbi',
        '%Pronomi'
        ]

    pos_tag_m = [
        'Travel Blog Maschi',
        lexical_richness(files[0]),
        ]
    pos_tag_m.extend(pos_tag(files[0], 'percentuali'))

    pos_tag_f = [
        'Travel Blog Femmine',
        lexical_richness(files[1]),
        ]
    pos_tag_f.extend(pos_tag(files[1], 'percentuali'))
    print tabulate([pos_tag_m, pos_tag_f], headers3)

    # TABELLA OUTPUT 4
    print '\n'
    headers4 = [
        'Corpora',
        'M sost/frase',
        'M agg/frase',
        'M verbi/frase',
        'M pron/frase'
    ]

    pos_medie_m = [
        'Travel Blog Maschi'
    ]
    pos_medie_m.extend(pos_tag(files[0], 'medie'))

    pos_medie_f = [
        'Travel Blog Femmine'
    ]
    pos_medie_f.extend(pos_tag(files[1], 'medie'))
    print tabulate([pos_medie_m, pos_medie_f], headers4)

# >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< ><
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
files = ['TBM.txt', 'TBF.txt']
# files = [sys.argv[0], sys.argv[1]]
tokenized = {
    'm':tokenized_text(files[0]),
    'f':tokenized_text(files[1])
    }
# >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< ><
main()