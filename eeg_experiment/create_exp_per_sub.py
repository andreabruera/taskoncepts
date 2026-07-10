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
         'Geräusche',
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
triggers_dims = {tuple(n) : i+100 for i, n in enumerate(dim_ques)}

for sub in range(41):
    sub_bids = 'sub-{:02}'.format(sub)
    rand_dimques = random.sample(dim_ques, k=len(dim_ques))
    for run, dq in enumerate(rand_dimques):
        run_bids = 'run-{:02}'.format(run+1)
        dim = dq[0]
        ques = dq[1]
        keys = '[1=niedrig      2=mittel      3=hoch]' if run%2==0 else '[1=hoch      2=mittel      3=niedrig]'
        run_words = random.sample(words, k=len(words))
        fold = os.path.join('randomized_exp_per_sub', sub_bids, run_bids)
        os.makedirs(fold, exist_ok=True)
        with open(os.path.join(fold, '{}_{}_dimension-question.tsv'.format(sub_bids, run_bids)), 'w') as o:
            o.write('{}\t{}\t{}\t{}\n'.format(dim, ques, keys, triggers_dims[tuple(dq)]))
        with open(os.path.join(fold, '{}_{}_words.tsv'.format(sub_bids, run_bids)), 'w') as o:
            for w in run_words:
                o.write('{}\t{}\n'.format(w, triggers_nouns[w]))
            o.write('\n')
        fold = os.path.join('experiment_logs', sub_bids, run_bids)
        os.makedirs(fold, exist_ok=True)
