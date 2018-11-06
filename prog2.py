# -*- coding: utf-8 -*-

import sys, codecs, nltk, re, math
from nltk import FreqDist, bigrams, trigrams
from tabulate import tabulate # libreria per output tabulari

class Corpus: # creo un elemento di tipo Corpus, per ciasuno dei quali faccio le dovute analisi
    def __init__ (self, path, name):
        self.path = path # percorso del file
        self.name = name # nome del corpus
        self.pos_tag = [] # [ (part_of_speech, token) ]
        self.tokens = self.tokenized_text() # lista dei token
        self.top20 = {
            'noPunct': self.top20_no_punct(),
            'adj'    : self.top20_partsofspeech('JJ'),
            'verb'   : self.top20_partsofspeech('V'),
            'noun'   : self.top20_partsofspeech('NN'),
        }
        self.top10tags = self.t10t()
        self.top10trigrams = FreqDist(trigrams([b for (a, b) in self.pos_tag])).most_common(10)
        self.bigram_data = {
            'bigrams': list(bigrams(self.tokens)),
            'unique' : None, # lista di bigrammi senza duplicati
            'condit' : [],   # probabilità condizionata
            'joined' : [],   # probabilità congiunta
            'fdist'  : None, # distribuzione di frequenza dei bigrammi
        }
        self.bigram_data['unique'] = set(self.bigram_data['bigrams'])
        self.bigram_prob() # inizializza self.bigram_data['condit'] & self.bigram_data['joined']

    def tokenized_text(self):
        # T O K E N I Z Z A
        corpora = codecs.open(self.path, "r", "utf-8")
        raw = corpora.read()
        phrases = sent_tokenizer.tokenize(raw.lower())
        tokenList = []
        for p in phrases:
            tokens = nltk.word_tokenize(p)
            tokenList += tokens
        corpora.close()
        self.pos_tag = nltk.pos_tag(tokenList)
        return tokenList

    def top20_no_punct(self):
        punct = re.compile(r'[^\w]') # lista di tutti i segni di punteggiatura
        tokenlist_noPunct = [i for i in self.tokens if not punct.match(i)]
        return FreqDist(tokenlist_noPunct).most_common(20)

    def top20_partsofspeech(self, t):
        a = []
        for token, tag in self.pos_tag:
            if tag.startswith(t):
                a.append(token)
        return FreqDist(a).most_common(20)

    def t10t(self):
        tuples = FreqDist(self.pos_tag).most_common()
        return nltk.FreqDist([b for (a, b), c in tuples]).most_common(10)
        
    def bigram_prob(self):
        # C O N D I T I O N A L
        for bigram in self.bigram_data['unique']:
            bigram_freq = self.bigram_data['bigrams'].count(bigram) # quante volte occorre un bigramma
            item1_freq = self.tokens.count(bigram[0]) # quante volte occorre il primo elemento
            prob_condit = bigram_freq*1.0 / item1_freq*1.0 # probabilità condizionata
            self.bigram_data['condit'].append([bigram, prob_condit]) # conservo ogni probabilità
        self.bigram_data['condit'].sort(key=getKey, reverse=True) # ordinamento decrescente
        self.bigram_data['condit'] = self.bigram_data['condit'][:10] # stacco le 10 più frequenti
        # J O I N E D
        pos_bigrams = list(bigrams([p for t, p in self.pos_tag]))
        pos_bigrams_fdist = nltk.FreqDist(pos_bigrams)
        pos_bigrams_set = set(pos_bigrams)
        for b in pos_bigrams_set:
            prob_joined = pos_bigrams_fdist[b]*1.0/len(self.tokens)*1.0
            self.bigram_data['joined'].append([b, prob_joined])
        self.bigram_data['joined'].sort(key=getKey, reverse=True)
        self.bigram_data['joined'] = self.bigram_data['joined'][:10]

# S U P P O R T O

def getKey(e): # per ordinare in base al secondo elemento di coppie
    return e[1]
    
def log2(x): # per calcolare log in base 2
    return math.log(x)/math.log(2)

def main():
    #  M E N U - S E L E Z I O N E
    # choice = input('Premi:\n\
    # 1 per top 20 token punteggiatura esclusa\n\
    # 2 per top 20 aggettivi\n\
    # 3 per top 20 verbi\n\
    # 4 per top 10 PoS\n\
    # 5 per top 10 trigrammi\n\
    # 6 per probabilità congiunta\n\
    # 7 per probabilità condizionata\n\
    # 8 per top 10 sostantivi + agg.\n\
    # 9 TBD\n\
    # 0 per uscire\n')
    choice = 8

    while choice != 0:
        if choice == 1: # tabella senza punteggiatura
            headers = ['top 20 token\n'+m.name, 'freq.', 'top 20 token\n'+f.name, 'freq.']
            records = []
            for (a, b), (c, d) in zip(m.top20['noPunct'],f.top20['noPunct']):
                records.append([a, b, c, d])
            print '\n', tabulate(records, headers)
        elif choice == 2: # tabella aggettivi
            headers = ['top 20 AGGETTIVI\n'+m.name, 'freq.', 'top 20 AGGETTIVI\n'+f.name, 'freq.']
            records = []
            for (a, b), (c, d) in zip(m.top20['adj'],f.top20['adj']):
                records.append([a, b, c, d])
            print '\n', tabulate(records, headers)
        elif choice == 3: # tabella verbi
            headers = ['top 20 VERBI\n'+m.name, 'freq.', 'top 20 VERBI\n'+f.name, 'freq.']
            records = []
            for (a, b), (c, d) in zip(m.top20['verb'],f.top20['verb']):
                records.append([a, b, c, d])
            print '\n', tabulate(records, headers)
        elif choice == 4: # tabella top 10 parts of speech
            headers = ['top 10 POS_tags\n'+m.name, 'freq.', 'top 10 POS_tags\n'+f.name, 'freq.']
            records = []
            for (a, b), (c, d) in zip(m.top10tags, f.top10tags):
                records.append([a, b, c, d])
            print '\n', tabulate(records, headers)
        elif choice == 5: # tabella trigrammi
            headers = ['top 10 trigrams\n'+m.name, 'freq.', 'top 10 trigrams\n'+f.name, 'freq.']
            records = []
            for ((onem, twom, threem), freqm),((onef, twof, threef), freqf) in zip(m.top10trigrams, f.top10trigrams):
                records.append([(onem, twom, threem), freqm, (onef, twof, threef), freqf])
            print '\n', tabulate(records, headers)
        elif choice == 6: # probabilità congiunta
            headers = ['MASCHI\nbigrammi', 'prob.\ncongiunta', 'FEMMINE\nbigrammi', 'prob.\ncongiunta']
            records = []
            for [bigram_a, freq_a], [bigram_b, freq_b] in zip(m.bigram_data['joined'],f.bigram_data['joined']):
                records.append([bigram_a, str(freq_a*100) + ' %', bigram_b, str(freq_b*100) + ' %'])
            print '\n', tabulate(records, headers)
            # NOTA: SISTEMARE OUTPUT
        elif choice == 7: # probabilità condizionata
            headers = ['MASCHI\nbigrammi', 'prob.\ncondizionata', 'FEMMINE\nbigrammi', 'prob.\ncondizionata']
            records = []
            for [bigram_a, freq_a], [bigram_b, freq_b] in zip(m.bigram_data['condit'],f.bigram_data['condit']):
                records.append([bigram_a, str(freq_a*100) + ' %', bigram_b, str(freq_b*100) + ' %'])
            print '\n', tabulate(records, headers)
        elif choice == 8: # sostantivi più frequenti combinati

            comb = { # combino elementi di m ed f
                'tokens' : [], # token combinati
                'tags'   : [], # pos_tag combinati
                'top20NN': [], # (sost, freq) combinati
                'nouns'  : [], # sost combinati
                'adj'    : [], # aggettivi combinati
                'fdistJJ': [], # fdist aggettivi combinati
                'bigrams': [], # bigrammi combinati
                'fdistBI': [], # fdist bigrammi combinati
                'unique' : [], # bigrammi senza duplicati
                'small'  : [], # lista contenente solo i bigrammi che mi interessano
            }
            comb['tokens'] = m.tokens
            comb['tokens'].extend(f.tokens)
            comb['tags'] = m.pos_tag
            comb['tags'].extend(f.pos_tag) 
            NN_list = []
            for token, tag in comb['tags']:
                if tag.startswith('NN'):
                    NN_list.append(token)
            comb['top20NN'] = FreqDist(NN_list).most_common(20)
            comb['nouns'] = [n for n, fre in comb['top20NN']]
            comb['adj'] = [token for token, tag in comb['tags'] if tag.startswith('JJ')]
            comb['fdist_JJ'] = nltk.FreqDist(comb['adj'])
            comb['bigrams'] = m.bigram_data['bigrams']
            comb['bigrams'].extend(f.bigram_data['bigrams'])
            comb['unique'] = set(comb['bigrams'])
            comb['small'] = [i for i in comb['unique'] if (i[1] in comb['nouns']) and (i[0] in comb['adj'])]
            comb['fdistBI'] = nltk.FreqDist(comb['small'])
            headers = ['aggettivo', 'local mutual\ninformation']
            for n in comb['nouns']: # per ogni sostantivo da analizzare
                LMI = 0.0
                records = [] # pulisco la tabella
                JJeLMI_Tuples = [] # azzero le tuple
                for b in comb['small']: # per ogni bigramma
                    if ((b[1]==n) and (b[0] in comb['adj'])): # se il sostantivo è quello che cerco
                        freq_NN = comb['tokens'].count(b[1])
                        freq_JJ = comb['tokens'].count(b[0])
                        freq_observed = comb['fdistBI'][b]
                        freq_expected = ((freq_JJ*1.0)*(freq_NN*1.0))/(len(comb['tokens'])*1.0)
                        LMI = (freq_observed*1.0)*math.log((freq_observed*1.0)/(freq_expected*1.0), 2)
            JJeLMI_Tuples.append((b[1], (b[0], int(LMI)))) # genero una tupla JJ + LMI
                JJeLMI_Tuples.sort(key=getKey, reverse=True) # ho finito; ordino le tuple x LMI
                # for e1, e2 in JJeLMI_Tuples:
                #     records.append([e1, e2])
                # print 'Sostantivo: '+ str(n[0]) +' - Occorrenze: '+ str(n[1])+'\n'
                # print tabulate(records, headers, floatfmt=".2f"), '\n'
                
                # NOTA: da capire perché vengono stampate in output tutti gli aggettivi per ogni
                # nome invece di stampare solo gli aggettivi accoppiati


        elif choice == 9: # 20 nomi propri di luogo più frequenti
            pass # ! da fare
        choice = 0 # uscita automatica
    return

# >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< ><
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
files = ['TBM.txt', 'TBF.txt']
m = Corpus(files[0], 'travel blog maschi')
f = Corpus(files[1], 'travel blog femmine')
main()

# [[(u'i', 106)], [(u'overall', 1510.0049659582799), (u'armenia', 1510.0049659582799),
# (u'travel', 1510.0049659582799), (u'else', 1404.0049659582799), (u'\u2014', 1341.9989408818374)], [(u'world', 62)], [(u'whichever', 883.2104517869184), (u'diverse', 821.2104517869184), (u'borderless', 821.2104517869184), (u'small', 668.725691431406), (u'different', 647.1544466193469), (u'new', 610.8867715746353)], [(u'class', 52)], [(u'yoga', 740.7571531116089), (u'capital-rich', 740.7571531116089), (u'labor-providing', 740.7571531116089), (u'crossfit', 740.7571531116089), (u'capital-owned', 740.7571531116089), (u'capitalistic', 688.7571531116089), (u'\u2018', 658.339103074109), (u'only', 560.8667089424696), (u'new', 512.3566471271134), (u'middle', 480.75715311160894)], [(u'country', 37)], [(u'land-locked', 527.0772050986449), (u'polarizing', 527.0772050986449), (u'regular', 490.0772050986448), (u'anglo-saxon', 490.0772050986448), (u'western', 468.4335925719621), (u'amazing', 453.0772050986448), (u'exotic', 423.2050729825135), (u'only', 399.07823520906487), (u'first', 390.1609355274244), (u'other', 340.4346226823821)], [(u'people', 37)], [(u'insane', 527.0772050986449), (u'regular', 490.0772050986448), (u'happy', 441.1658655878125), (u'japanese', 404.16586558781245), (u'most', 390.1609355274244), (u'much', 382.5222530611297), (u'many', 349.2050729825135), (u'other', 340.4346226823821), (u'more', 332.90388710123216)], [(u'time', 37)], [(u'wait', 527.0772050986449), (u'precious', 490.0772050986448), (u'biggest', 468.4335925719621), (u'same', 468.4335925719621), (u'perfect', 441.1658655878125), (u'hard', 441.1658655878125), (u'next', 404.16586558781245), (u'first', 390.1609355274244), (u'much', 382.5222530611297)], [(u'capital', 36)], [(u'invisible', 476.8318752311139), (u'political', 476.8318752311139), (u'nomad', 455.77322520515236), (u'big', 393.2424638151689), (u'old', 375.76709803704017), (u'new', 354.70844801107853), (u'own', 336.18381378920725)], [(u'things', 30)], [(u'do', 427.35989602592826), (u'certain', 427.35989602592826), (u'sell', 427.35989602592826), (u'useless', 397.35989602592826), (u'stupid', 379.8110210042936), (u'amazing', 367.35989602592826), (u'top', 367.35989602592826), (u'weird', 357.7020531793074), (u'only', 323.5769474668093), (u'great', 313.1392483642001), (u'few', 293.5769474668093), (u'best', 286.3467044816955)], [(u'place', 28)], [(u'inexpensive', 398.86923629086635), (u'spooky', 398.86923629086635), (u'kid-friendly', 370.86923629086635), (u'special', 354.490286270674), (u'fantastic', 354.490286270674), (u'nice', 354.490286270674), (u'hard', 333.85524963402025), (u'awesome', 333.85524963402025), (u'popular', 314.86923629086635), (u'beautiful', 310.11133625048166), (u'good', 286.86923629086635)], [(u'tokyo', 27)], [(u'cafes', 384.6239064233354),
# (u'downtown', 384.6239064233354), (u'lockup', 384.6239064233354), (u'central', 321.9318478613767)], [(u'family', 27)], [(u'rica', 384.6239064233354), (u'wonderful', 384.6239064233354), (u'entire', 330.6239064233354), (u'nicaragua', 330.6239064233354), (u'whole', 330.6239064233354)], [(u'way', 26)], [(u'fun', 370.37857655580444), (u'milky', 370.37857655580444), (u'bedouin', 370.37857655580444), (u'platonic', 370.37857655580444), (u'efficient', 344.37857655580444), (u'american', 297.3873485823068), (u'unique', 287.9605265183044), (u'great', 271.3873485823068), (u'different', 271.3873485823068), (u'best', 248.16714388413612)], [(u'day', 26)], [(u'freaking', 370.37857655580444), (u'1-2', 370.37857655580444), (u'3-4', 370.37857655580444), (u'clear', 329.1695515370545), (u'nice', 329.1695515370545), (u'popular', 292.37857655580444), (u'next', 284.0084460887331), (u'other', 239.2243294524847)], [(u'life', 25)], [(u'daily', 331.1332466882735), (u'real', 285.94937363683346), (u'incredible', 276.8851216522158), (u'armenian', 258.4609817980606), (u'good', 256.1332466882735)], [(u'city', 23)], [(u'mexico', 327.64258695321166), (u'pleasant', 327.64258695321166), (u'cheapest', 327.64258695321166), (u'largest', 304.64258695321166), (u'western', 291.1884494366251), (u'japanese', 251.23824077080232), (u'different', 240.07342374588677)], [(u'money', 23)],
# [(u'asset\u2014time\u2014for', 327.64258695321166), (u'enough', 304.64258695321166),
# (u'such', 251.23824077080232)], [(u'america', 22)], [(u'latin', 269.39725708568074),
# (u'central', 262.3148389981588)], [(u'japan', 21)], [], [(u'years', 21)], [(u'ten', 299.15192721814975), (u'recent', 299.15192721814975), (u'light', 278.15192721814975),
# (u'few', 205.50386322676655), (u'many', 198.19747385494009)],


# [(u'days', 20)], 
# [(u'lazy', 233.20734733619574), (u'most', 210.89780298779698), (u'few', 195.7179649778729), (u'other', 184.01871496344978)]]