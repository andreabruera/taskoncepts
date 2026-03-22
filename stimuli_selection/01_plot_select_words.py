import matplotlib
import numpy
import os
import pickle

from matplotlib import font_manager, pyplot

font_folder = os.path.join('..', '..', 'fonts')
if os.path.exists(font_folder):
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

de_to_en = dict()
ws_f = os.path.join('..', 'data', 'de', 'words.csv')
ws = list()
cats = dict()
with open(ws_f) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split(',')
        w = line[0]
        ws.append(w)
        cat = (line[-1], line[-2])
        cats[w] = cat
        de_to_en[line[0]] = line[1]
assert len(ws) == 535

### reading concreteness, familiarity and imageability

other_norms = numpy.zeros(shape=(535, 3))

orig_others = ['Concreteness', 'Familiarity', 'Imageability']

f = os.path.join('..', 'data', 'de', 'word_attribute_means_wide_EN.csv')
with open(f) as i:
    for l_i, l in enumerate(i):
        line = [w.strip() for w in l.strip().split(',')]
        if l_i == 0:
            header = line.copy()
            assert len(header) == 52
            others = [(h_i, h) for h_i, h in enumerate(header) if h in orig_others and h_i>0]
            assert [v[1] for v in others] == orig_others
            assert len(others) == 3
            continue
        try:
            idx = ws.index(line[0])
        except ValueError:
            idx = ws.index(line[0].lower())
        other_norms[idx] = numpy.array([line[h_i] for h_i, h in others], dtype=numpy.float32)

chosen_conc = [
        ('living object', 'animal'),
        #('living object', 'plant'),
        #('artifact', 'tool'),
        ('artifact', 'food'),
        ('artifact', 'place'),
        ('artifact', 'instrument')
               #('living object', 'human'), ### very good
               #('artifact', 'place'), ### not that good
               #('living object', 'animal'),
               #('living object', 'plant'),
               #('artifact', 'tool'),
               #('natural object', 'place'),
               #('artifact', 'instrument'),  ### not good!
               #('artifact', 'vehicle'),
               #('artifact', 'food'), ### not good
               ]
chosen_abs = [('abstract entity', 'time period'), ('mental entity', 'emotion'), ('abstract entity', 'abstract construct'), ('abstract entity', 'social construct')]

### opensubs frequencies
os_f = os.path.join('pkls', 'de_opensubs_cased_word_freqs.pkl')
with open(os_f, 'rb') as i:
    os_freqs = pickle.load(i)
### wac frequencies
wf_f = os.path.join('pkls', 'de_wac_cased_word_freqs.pkl')
with open(wf_f, 'rb') as i:
    wac_freqs = pickle.load(i)
### wac old20
wo_f = os.path.join('pkls', 'de_wac_10_min-uncased_OLD20.pkl')
with open(wo_f, 'rb') as i:
    old = pickle.load(i)

w_stats = numpy.zeros(shape=(len(ws), 6))
### length, opensubs freq, wac freq, wac old20
for w_i, w in enumerate(ws):
    ### filtering for length
    #if w in ['Rallye', 'Grill', 'Zirkus', 'Parade', 'Opfer', 'Autor', 'Zeuge', 'Laster']:
    #    continue
    if w in ['Geschäft', 'Medizin',]:
        continue
    #if 'abstract' not in cats[w][0] and 'mental' not in cats[w][0]:
    if cats[w] not in chosen_abs and cats[w] not in chosen_conc:
        continue
    #if 'abstract' in cats[w][0] or 'mental' in cats[w][0]:
    #    continue
    length = len(w)
    if not w[0].isupper():
        continue
    if other_norms[w_i, 1] < 2.:
        continue
    try:
        freq = numpy.log10(wac_freqs[w])
    except KeyError:
        #print(w)
        continue
    try:
        os_freq = numpy.log10(os_freqs[w])
    except KeyError:
        continue
    ### filtering for freq
    #if freq < 2.5 or freq > 5:
    #    continue
    #if os_freq < 2. or os_freq >= 5.:
    #    continue
    if os_freq < 2.:
        continue
    #if length < 5 or length > 7:
    #    continue
    if length < 5 or length > 8:
        continue
    w_old = old[w.lower()]
    #if w_old <= 1.5 or w_old > 3.:
    #if w_old <= 1.5:
    #    continue
    w_stats[w_i, 0] = length
    w_stats[w_i, 1] = os_freq
    #w_stats[w_i, 2] = freq
    w_stats[w_i, 2] = w_old
    w_stats[w_i, 3] = other_norms[w_i, 1]
    w_stats[w_i, 4] = other_norms[w_i, 0]
    w_stats[w_i, 5] = other_norms[w_i, 2]

mask = w_stats[:, 0]>0.
w_stats = w_stats[mask]
sel_ws = numpy.array(ws)[mask].tolist()

base_folder = os.path.join('plots', 'first_selection')
os.makedirs(base_folder, exist_ok=True)
for idx, case in enumerate(['word-length', 'opensubs-frequency', 'wac-old20', 'familiarity', 'concreteness', 'imageability']):
    fig, ax = pyplot.subplots(
                              constrained_layout=True,
                              )
    for curr_cats in [chosen_conc, chosen_abs]:
        w_idxs = [_ for _, w in enumerate(sel_ws) if cats[w] in curr_cats]
        ax.hist(w_stats[w_idxs, idx], bins=50, alpha=0.5)
    out_f = os.path.join(base_folder, '{}.jpg'.format(case))
    pyplot.savefig(out_f)
    pyplot.clf()
    pyplot.close()

print(len(sel_ws))
sel_cats = {k : list() for k in cats.values()}
for w in sel_ws:
    w = str(w)
    try:
        sel_cats[cats[w]].append(w)
    except KeyError:
        sel_cats[cats[w]] = [w]

mapper = {
          #'concrete' : ['living object', 'artifact', 'natural object', 'physical action', 'physical state', 'physical property'],
          #'abstract' : ['event', 'abstract entity', 'mental entity', 'abstract action', 'mental state', 'abstract property', ],
          #'abstract' : check,
          'concrete' : [k[0] for k in chosen_conc],
          'abstract' : [k[0] for k in chosen_abs],
          }
inv_mapper = {k : v for v, _ in mapper.items() for k in _}
abs_con = {k : dict() for k in mapper.keys()}

gen_avg_freq = numpy.average([os_freqs[w] for w in sel_ws])
gen_avg_fam = numpy.average([w_stats[sel_ws.index(w), 3] for w in sel_ws])
gen_min_fam = min([w_stats[sel_ws.index(w), 3] for w in sel_ws])
gen_max_fam = max([w_stats[sel_ws.index(w), 3] for w in sel_ws])
gen_min_freq = min([os_freqs[w] for w in sel_ws])
gen_max_freq = max([os_freqs[w] for w in sel_ws])


for k, v in sel_cats.items():
    try:
        assert len(v) >= 8
        #print(k)
    except AssertionError:
        #print(k)
        continue
    abscon_v = inv_mapper[k[0]]
    #if abscon_v == 'concrete':
    #    sorted_v = sorted(v, key=lambda item : abs(6-len(item)))[:5]
    #else:
    #    sorted_v = sorted(v, key=lambda item : abs(6-len(item))-(numpy.log(len(item))), reverse=True)[:5]
    if abscon_v == 'concrete':
        #sorted_v = sorted(v, key=lambda item : abs(gen_avg_freq-wac_freqs[item]))[:5]
        #sorted_v = sorted(v, key=lambda item : abs(gen_max_freq-os_freqs[item]))[:8]
        #sorted_v = sorted(v, key=lambda item : abs(gen_avg_freq-wac_freqs[item])-(wac_freqs[item])+(w_stats[sel_ws.index(item), 5])**2)[:5]
        #sorted_v = sorted(v, key=lambda item : abs(gen_avg_freq-wac_freqs[item])+(abs(gen_min_freq-wac_freqs[item])*0.25)+w_stats[sel_ws.index(item), 5])[:5]
        #sorted_v = sorted(v, key=lambda item : (abs(gen_avg_freq-wac_freqs[item])*0.01)+abs(gen_min_fam-w_stats[sel_ws.index(item), 5]))[:5]
        sorted_v = sorted(v, key=lambda item : abs(gen_min_fam-w_stats[sel_ws.index(item), 3]))[:8]
    else:
        sorted_v = sorted(v, key=lambda item : abs(gen_max_fam-w_stats[sel_ws.index(item), 3]))[:8]
        #sorted_v = sorted(v, key=lambda item : abs(wac_freqs[item]-gen_min_freq))[:5]
        #sorted_v = sorted(v, key=lambda item : abs(wac_freqs[item]-gen_min_freq))[:5]
        #sorted_v = v[:5]
        #sorted_v = sorted(v, key=lambda item : abs(gen_min_fam-w_stats[sel_ws.index(item), 5]))[:5]
        #sorted_v = sorted(v, key=lambda item : abs(gen_avg_fam-w_stats[sel_ws.index(item), 5]))[:5]
    if abscon_v == 'concrete' and k not in chosen_conc:
        print(k)
        continue
    abs_con[abscon_v][k] = sorted_v
### selecting 64 words
assert len([w for _ in abs_con.values() for ws in _.values() for w in ws]) == 64
base_folder = os.path.join('plots', 'second_selection')
os.makedirs(base_folder, exist_ok=True)
#for idx, case in enumerate(['word-length', 'opensubs-frequency', 'wac-frequency', 'wac-old20']):
cases = ['Word\nlength', 'Word frequency\n(Opensubtitles)','OLD20\n(DeWac)', 'Familiarity', 'Concreteness', 'Imageability']
fig, ax = pyplot.subplots(
                          constrained_layout=True,
                          figsize=(20, 10)
                          )
with open('german_selected_words.tsv', 'w') as o:
    o.write('concreteness\tfine_cat\tword_de\tword_en\tword_length\tlog10_freq_opensubtitles\tlog10_freq_wac\told20_wac\tconcreteness_rating\tfamiliarity\timageability\n')
    for ac in abs_con.keys():
        for cat, ws in abs_con[ac].items():
            for w in ws:
                conc = 0 if ac=='abstract' else 1
                o.write('{}\t'.format(conc))
                o.write('{}\t'.format(cat[1].replace(' ', '_')))
                o.write('{}\t'.format(w))
                o.write('{}\t'.format(de_to_en[w]))
                o.write('{}\t'.format(len(w)))
                idx = sel_ws.index(w)
                o.write('{}\t'.format(w_stats[idx, 1]))
                o.write('{}\t'.format(wac_freqs[w]))
                o.write('{}\t'.format(w_stats[idx, 2]))
                o.write('{}\t'.format(w_stats[idx, 3]))
                o.write('{}\t'.format(w_stats[idx, 4]))
                o.write('{}\n'.format(w_stats[idx, 5]))

        side = 'low' if ac=='concrete' else 'high'
        corr = -.1 if ac=='concrete' else .1
        second_mask = sorted([sel_ws.index(w) for ws in abs_con[ac].values() for w in ws])
        second_stats = w_stats[second_mask]
        parts = ax.violinplot(second_stats, positions=[x+corr for x in range(6)], side=side)
        pyplot.scatter(
                       [x+corr for x in range(6)],
                       numpy.average(second_stats, axis=0),
                       color='white',
                       zorder=2.5,
                       s=1000,
                       )
        pyplot.scatter(
                       [x+corr for x in range(6)],
                       numpy.average(second_stats, axis=0),
                       #color=colors[g_i],
                       zorder=3.,
                       s=500,
                       alpha=0.3,
                       label=ac,
                       )
        for pc in parts['bodies']:
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
ax.legend(fontsize=30, ncols=2, loc=8)
pyplot.xticks(ticks=range(6), labels=cases, fontsize=25, fontweight='bold')
pyplot.yticks(fontsize=20)
pyplot.ylabel('Log10 frequency / Word length / OLD20 / Rating', fontsize=20, fontweight='bold')
out_f = os.path.join(base_folder, 'stimuli_selection.jpg'.format(case))
pyplot.savefig(out_f)
pyplot.clf()
pyplot.close()

