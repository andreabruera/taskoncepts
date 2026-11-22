import os
import random

norms_en = [
         'Audition',
         'Taste',
         'UpperLimb',
         'Touch',
         'Smell',
         'Vision',
         'Arousal',
         'Caused',
         'Needs',
         'Harm',
         'Communication',
         'Time',
           ]

norms_de = [
         'Hören',
         'Geschmack',
         'Arm-/Handbewegungen',
         'Berührung',
         'Geruch',
         'Visuell',
         'Erregung',
         'Verursacht',
         'Bedürfnisse',
         'Schaden',
         'Kommunikation',
         'Zeit',
         'Vertrautheit',
           ]
### only used the first time
#lines = set()
#with open(os.path.join('..', '..', '..', 'chunks_all.tsv')) as i:
#    for l_i, l in enumerate(i):
#        if l_i == 0:
#            continue
#        line = tuple([lin.strip() for lin in l.strip().split('\t')][2:4])
#        if l.split('\t')[1] != '1':
#            continue
#        lines.add(line)
#for norm in norms_de:
#    assert norm in [v[0] for v in lines]
#with open('dimensions_questions.tsv', 'w') as o:
#    for dim, ques in sorted(lines, key = lambda item : item[0]):
#        o.write('{}\t{}\n'.format(dim, ques))
dim_ques = list()
with open('dimensions_questions.tsv') as o:
    for l in o:
        dim_ques.append(l.strip().split('\t'))
for norm in norms_de:
    assert norm in [v[0] for v in dim_ques]
dim_ques = [k for k in dim_ques if k[0] in norms_de]
words = list()
with open(os.path.join('..', 'stimuli_selection', 'german_selected_words.tsv')) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        w = l.split('\t')[2]
        words.append(w)
print(len(words))
triggers_nouns = {n : i+10 for i, n in enumerate(words)}
with open('triggers_nouns.tsv', 'w') as o:
    for w, trigger in triggers_nouns.items():
        o.write('{}\t{}\n'.format(w, trigger))
'''
Arm-/Handbewegungen	Wie stark denken Sie bei diesem Wort an eine Handlung mit Händen, Armen oder Fingern?	100	sensorimotor
Bedürfnisse	Wie stark denken Sie bei diesem Wort an etwas, ohne das Sie nur schwer leben könnten?	101	non-sensorimotor
Berührung	Wie stark denken Sie bei diesem Wort an etwas, das man durch Berührung gut erkennen kann?	102	sensorimotor
Erregung	Wie stark denken Sie bei diesem Wort an etwas, das Sie stark aktiviert oder erregt – egal ob positiv oder negativ?	103	non-sensorimotor
Geruch	Wie stark denken Sie bei diesem Wort an etwas mit einem bestimmten Geruch?	104	sensorimotor
Geräusche	Wie stark denken Sie bei diesem Wort an ein typisches oder auffälliges Geräusch?	105	sensorimotor
Geschmack	Wie stark denken Sie bei diesem Wort an etwas mit einem bestimmten Geschmack?	106	sensorimotor
Hören	Wie stark denken Sie bei diesem Wort an etwas, das man gut hören kann?	107	sensorimotor
Kommunikation	Wie stark denken Sie bei diesem Wort an etwas, das zur Kommunikation zwischen Menschen dient?	108	non-sensorimotor
Schaden	Wie stark denken Sie bei diesem Wort an etwas, das Ihnen oder anderen schaden könnte?	109	non-sensorimotor
Vertrautheit	Wie vertraut ist Ihnen diese Sache?	110	familiarity
Verursacht	Wie stark denken Sie bei diesem Wort an etwas, das durch ein bestimmtes Ereignis oder eine Handlung verursacht wird?	111	non-sensorimotor
Zeit	Wie stark denken Sie bei diesem Wort an etwas, das zu einer bestimmten Zeit passiert?	112	non-sensorimotor
'''
triggers_dims = {tuple(n) : i+100 for i, n in enumerate(dim_ques)}
with open('triggers_dimques.tsv', 'w') as o:
    for w, trigger in triggers_dims.items():
        o.write('{}\t{}\t{}\n'.format(w[0], w[1], trigger))

fold = os.path.join('randomized_exp_per_sub_05082026',)
os.makedirs(fold, exist_ok=True)

random.seed(12)
for sub in range(18, 41):
    sub_bids = 'sub-{:02}'.format(sub)
    rand_dimques = random.sample(dim_ques, k=len(dim_ques))
    for run, dq in enumerate(rand_dimques):
        run += 1
        run_bids = 'run-{:02}'.format(run+1)
        dim = dq[0]
        ques = dq[1]
        keys = '[1=niedrig      2=mittel      3=hoch]' if run%2==0 else '[1=hoch      2=mittel      3=niedrig]'
        run_words = random.sample(words, k=len(words))
        #with open(os.path.join(fold, '{}_{}_dimension-question.events'.format(sub_bids, run_bids)), 'w') as o:
        with open(os.path.join(fold, '{:02}_{:02}_dq.mp3'.format(sub, run)), 'w') as o:
            o.write('Var1\tVar2\tVar3\tVar4\n')
            o.write('{}\t{}\t{}\t{}\n'.format(dim, ques, keys, triggers_dims[tuple(dq)]))
        with open(os.path.join(fold, '{:02}_{:02}_w.mp3'.format(sub, run)), 'w') as o:
            o.write('Var1\tVar2\n')
            for w in run_words:
                o.write('{}\t{}\n'.format(w, triggers_nouns[w]))
            o.write('\n')
