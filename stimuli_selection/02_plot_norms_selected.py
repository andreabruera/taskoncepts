import matplotlib
import numpy
import os
import random
import sklearn

from matplotlib import font_manager, pyplot
from sklearn import manifold

os.makedirs(os.path.join('plots', 'norms_selected', 'full_dims'), exist_ok=True)
os.makedirs(os.path.join('plots', 'norms_selected', 'reduced_dims'), exist_ok=True)

### interpreting dimensions
### read dimensions
dim_cat = dict()
f = os.path.join('..', 'data', 'en', 'binder_sections.tsv')
with open(f) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i <= 32:
            ac = 'concrete'
        else:
            ac = 'abstract'
        dim_cat[line[1]] = (line[0], ac)
assert len(dim_cat.keys()) == 65

font_folder = os.path.join('..', '..', 'fonts')
if os.path.exists(font_folder):
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

abstract_cols = ['mediumorchid', 'plum', 'violet', 'thistle', 'blueviolet', 'slateblue']
concrete_cols = ['teal', 'lightseagreen', 'mediumturquoise', 'aquamarine', 'paleturquoise', 'mediumseagreen']

color_mapper = dict()
abs_con = {'1' : list(), '0' : list()}

### german words
f = 'german_selected_words.tsv'
de_ws = list()
cats = dict()
inv_cats = dict()
with open(f) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        de_ws.append(line[2])
        cats[line[2]] = line[1]
        try:
            inv_cats[line[1]].append(line[2])
        except KeyError:
            inv_cats[line[1]] = [line[2]]
        if line[1] not in abs_con[line[0]]:
            abs_con[line[0]].append(line[1])
        if line[0] == '1':
            color_mapper[line[1]] = concrete_cols[abs_con[line[0]].index(line[1])]
        elif line[0] == '0':
            color_mapper[line[1]] = abstract_cols[abs_con[line[0]].index(line[1])]

assert len(de_ws) == 64

binder_norms = numpy.zeros(shape=(64, 48,))
other_norms = numpy.zeros(shape=(64, 3))

others = ['Concreteness', 'Familiarity', 'Imageability']

f = os.path.join('..', 'data', 'de', 'word_attribute_means_wide_EN.csv')
with open(f) as i:
    for l_i, l in enumerate(i):
        line = [w.strip() for w in l.strip().split(',')]
        if l_i == 0:
            header = line.copy()
            assert len(header) == 52
            binders = [(h_i, h) for h_i, h in enumerate(header) if h not in others and h_i>0]
            assert len(binders) == 48
            continue
        if line[0] not in de_ws:
            continue
        idx = de_ws.index(line[0])
        binder_norms[idx] = numpy.array([line[h_i] for h_i, h in binders], dtype=numpy.float32)

t_sne = manifold.TSNE(
    n_components=2,
    perplexity=8,
)
binder_norms_tsne = t_sne.fit_transform(binder_norms)
fig, ax = pyplot.subplots(
                          figsize=(20, 20),
                          )
for cat, ws in inv_cats.items():
    color = color_mapper[cat]
    ws_idxs = [_ for _, w in enumerate(de_ws) if w in ws]
    ax.scatter(
        binder_norms_tsne[ws_idxs, 0],
        binder_norms_tsne[ws_idxs, 1],
        color=color,
        s=750,
        linewidths=2.5,
        edgecolors='white',
           )
    for idx in ws_idxs:
        ax.text(
                x=binder_norms_tsne[idx, 0]+0.25,
                y=binder_norms_tsne[idx, 1],
                s=de_ws[idx],
                fontsize=20,
                ha='left',
                )
ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
pyplot.xticks(ticks=[])
pyplot.yticks(ticks=[])
pyplot.savefig(
               os.path.join('plots', 'norms_selected', 'full_dims',
               'tsne.jpg'),
               )

### reaction times
#Word_DE,Word_EN,Attribute_DE,Attribute_EN,rating,rating_Binder,rating_RT,prolific_ID,data_filename,ratings_filename,dataset
ratings = dict()
f = os.path.join('..', 'data', 'de', 'ratings_table_all_excludedDuplicates_V2.csv')
with open(f) as i:
    for l_i, l in enumerate(i):
        line = [w.strip() for w in l.strip().split(',')]
        if l_i == 0:
            header = line.copy()
            continue
        w = line[header.index('Word_DE')]
        if w not in de_ws:
            continue
        dim = line[header.index('Attribute_EN')]
        try:
            assert dim in [v[1] for v in binders]
            assert dim in dim_cat.keys()
        except AssertionError:
            assert dim in others
            continue
        rating = int(line[header.index('rating')])
        rt = float(line[header.index('rating_RT')])
        sub = line[header.index('prolific_ID')]
        try:
            ratings[sub].append((w, dim, rating, rt))
        except KeyError:
            ratings[sub] = [(w, dim, rating, rt)]

avg_std = dict()
for sub, sub_data in ratings.items():
    avg = numpy.average([v[-1] for v in sub_data])
    std = numpy.std([v[-1] for v in sub_data])
    avg_std[sub] = (avg, std)

rts = numpy.zeros(shape=(64, len(ratings.keys()), 48))
rts[rts==0.] = numpy.nan
z_rts = numpy.zeros(shape=(64, len(ratings.keys()), 48))
z_rts[z_rts==0.] = numpy.nan

subs = sorted(ratings.keys())
for sub, sub_data in ratings.items():
    for w, dim, rating, rt in sub_data:
        x = de_ws.index(w)
        y = subs.index(sub)
        z = [v[1] for v in binders].index(dim)
        rts[x, y, z] = rt
        z_rts[x, y, z] = (rt-avg_std[sub][0])/avg_std[sub][1]

### plotting by word
fig, ax = pyplot.subplots(
                          figsize=(24, 10),
                          )
data = numpy.nanmean(z_rts, axis=2)
assert data.shape == (64, len(subs))
for x in range(data.shape[0]):
    non_nans = data[x, :][~numpy.isnan(data[x, :])]
    parts = ax.violinplot(
              non_nans,
              positions=[x],
              side='low',
              )
    ax.scatter(
               x=x,
               y=numpy.average(non_nans),
               s=200,
               )
pyplot.vlines(
              x=range(64),
              ymin=-0.6,
              ymax=1.,
              alpha=0.3,
              linestyle='dashed',
              color='white',
              )
pyplot.hlines(
              y=0.,
              xmin=0.,
              xmax=64,
              alpha=0.5,
              color='black',
              )
pyplot.xticks(
              ticks=range(64),
              labels=de_ws,
              rotation=45,
              )
pyplot.tight_layout()
pyplot.savefig(
               os.path.join('plots', 'norms_selected', 'full_dims',
                            'word_rts.jpg'
                            ),
               pad_inches=0,
               )

### plotting by abstract/concrete
fig, ax = pyplot.subplots(
                          figsize=(20, 20),
                          )
data = numpy.nanmean(z_rts, axis=(1, 2))
assert data.shape == (64, )
corr = {'0' : -.4, '1' : .4}
for ac, cats in abs_con.items():
    assert ac in ['1', '0']
    ac_idxs = list()
    #for cat, ws in cats.items():
    for cat in cats:
        ws = inv_cats[cat]
        cat_idxs = [de_ws.index(w) for w in ws]
        rand_corrs = [random.randrange(-10, 10)*0.01 for _ in range(len(cat_idxs))]
        ac_idxs.extend(cat_idxs)
        ax.scatter(
                   [0+(corr[ac]*0.5)+rand_corrs[_] for _ in range(len(cat_idxs))],
                   data[cat_idxs],
                   color=color_mapper[cat],
                   s=200,
                   linewidth=1,
                   edgecolor='white',
                   )
        for _, w_i in enumerate(cat_idxs):
            ax.text(
                    x=0+(corr[ac]*0.5)+rand_corrs[_],
                    y=data[w_i],
                    s=de_ws[w_i],
                    fontsize=15,
                    )
    parts = ax.violinplot(
              data[ac_idxs],
              positions=[0+corr[ac]],
              side='low' if ac=='0' else 'high',
              showextrema=False,
              )
    col = 'c' if ac=='1' else 'm'
    ax.scatter(
              [0+corr[ac]],
              numpy.nanmean(data[ac_idxs]),
              s=750,
              color=col,
               linewidth=10,
               edgecolor='white',
               marker='D',
              )
    for pc in parts['bodies']:
        pc.set_facecolor(col)
pyplot.hlines(
              y=0.,
              xmin=-.75,
              xmax=.75,
              color='black',
              alpha=0.6,
              )
pyplot.xticks(
              ticks=[-.5, .5],
              labels=['Abstract', 'Concrete'],
              fontsize=40,
              fontweight='bold',
              )
pyplot.yticks(
              fontsize=30,
              )
pyplot.ylabel(
              'Z-scored RT across all 48 dimensions and subjects',
              fontsize=30,
              fontweight='bold',
              )
pyplot.tight_layout()
pyplot.savefig(
               os.path.join('plots', 'norms_selected', 'full_dims',
               'cats_rts.jpg'),
               pad_inches=0,
               )

### raw plotting by abstract/concrete
fig, ax = pyplot.subplots(
                          figsize=(20, 20),
                          )
data = numpy.nanmean(rts, axis=(1, 2))
assert data.shape == (64, )
corr = {'0' : -.4, '1' : .4}
for ac, cats in abs_con.items():
    assert ac in ['1', '0']
    ac_idxs = list()
    #for cat, ws in cats.items():
    for cat in cats:
        ws = inv_cats[cat]
        cat_idxs = [de_ws.index(w) for w in ws]
        rand_corrs = [random.randrange(-10, 10)*0.01 for _ in range(len(cat_idxs))]
        ac_idxs.extend(cat_idxs)
        ax.scatter(
                   [0+(corr[ac]*0.5)+rand_corrs[_] for _ in range(len(cat_idxs))],
                   data[cat_idxs],
                   color=color_mapper[cat],
                   s=200,
                   linewidth=1,
                   edgecolor='white',
                   )
        for _, w_i in enumerate(cat_idxs):
            ax.text(
                    x=0+(corr[ac]*0.5)+rand_corrs[_],
                    y=data[w_i],
                    s=de_ws[w_i],
                    fontsize=15,
                    )
    parts = ax.violinplot(
              data[ac_idxs],
              positions=[0+corr[ac]],
              side='low' if ac=='0' else 'high',
              showextrema=False,
              )
    col = 'c' if ac=='1' else 'm'
    ax.scatter(
              [0+corr[ac]],
              numpy.nanmean(data[ac_idxs]),
              s=750,
              color=col,
               linewidth=10,
               edgecolor='white',
               marker='D',
              )
    for pc in parts['bodies']:
        pc.set_facecolor(col)
pyplot.xticks(
              ticks=[-.5, .5],
              labels=['Abstract', 'Concrete'],
              fontsize=40,
              fontweight='bold',
              )
pyplot.yticks(
              fontsize=30,
              )
pyplot.ylabel(
              'Z-scored RT across all 48 dimensions and subjects',
              fontsize=30,
              fontweight='bold',
              )
pyplot.tight_layout()
pyplot.savefig(
               os.path.join('plots', 'norms_selected', 'full_dims',
               'raw_cats_rts.jpg'),
               pad_inches=0,
               )

### plotting by abstract/concrete concepts, separately for abstract/concrete dimensions
fig, ax = pyplot.subplots(
                          figsize=(20, 20),
                          )
assert data.shape == (64, )
corr = {'abstract' : -.5, 'concrete' : .5}
for ac, cats in abs_con.items():
    assert ac in ['1', '0']
    #x = int(ac)
    x = -1 if ac=='0' else 1
    #for cat, ws in cats.items():
    for __, ac2 in enumerate(['concrete', 'abstract']):
        dim_idxs = [_ for _, d in enumerate(binders) if dim_cat[d[1]][1]==ac2]
        assert len(dim_idxs) in [27, 21]
        data = numpy.nanmean(z_rts[:, :, dim_idxs], axis=(1, 2))
        ac_idxs = list()
        for cat in cats:
            ws = inv_cats[cat]
            cat_idxs = [de_ws.index(w) for w in ws]
            rand_corrs = [random.randrange(-10, 10)*0.01 for _ in range(len(cat_idxs))]
            ac_idxs.extend(cat_idxs)
            ax.scatter(
                       [x+(corr[ac2]*0.66)+rand_corrs[_] for _ in range(len(cat_idxs))],
                       data[cat_idxs],
                       color=color_mapper[cat],
                       s=200,
                       linewidth=1,
                       edgecolor='white',
                       )
            for _, w_i in enumerate(cat_idxs):
                ax.text(
                        x=x+(corr[ac2]*0.66)+rand_corrs[_],
                        y=data[w_i],
                        s=de_ws[w_i],
                        fontsize=15,
                        )
        assert len(ac_idxs) == 32
        parts = ax.violinplot(
                  data[ac_idxs],
                  positions=[x+corr[ac2]],
                  side='low' if ac2=='abstract' else 'high',
                  showextrema=False,
                  )
        col = 'darkseagreen' if ac2=='concrete' else 'pink'
        if ac2 == 'concrete':
            label = 'concrete\ndimensions'
        else:
            label = 'abstract\ndimensions'
        if ac == '1':

            ax.scatter(
                  [x+corr[ac2]],
                  numpy.nanmean(data[ac_idxs]),
                  s=750,
                  color=col,
                   linewidth=10,
                   edgecolor='white',
                   marker='D',
                   label=label,
                  )
        else:
            ax.scatter(
                  [x+corr[ac2]],
                  numpy.nanmean(data[ac_idxs]),
                  s=750,
                  color=col,
                   linewidth=10,
                   edgecolor='white',
                   marker='D',
                  )
        for pc in parts['bodies']:
            pc.set_facecolor(col)
ax.legend(fontsize=30, ncols=2, loc=9)
pyplot.hlines(
              y=0.,
              xmin=-1.75,
              xmax=1.75,
              color='black',
              alpha=0.6,
              )
pyplot.xticks(
              ticks=[-1.4, 1.4],
              labels=['Abstract\nconcepts', 'Concrete\nconcepts'],
              fontsize=40,
              fontweight='bold',
              )
pyplot.yticks(
              fontsize=30,
              )
pyplot.ylabel(
              'Z-scored RT across all 48 dimensions and subjects',
              fontsize=30,
              fontweight='bold',
              )
pyplot.tight_layout()
pyplot.savefig(
               os.path.join('plots', 'norms_selected', 'full_dims',
               'cats_dims_rts.jpg'),
               pad_inches=0,
               )

### constructing the matrix

### first organizing the dimensions
inv_dim_cat = dict()
for dim, vs in dim_cat.items():
    coarse = vs[0]
    ac = vs[1]
    if ac not in inv_dim_cat.keys():
        inv_dim_cat[ac] = dict()
    if coarse not in inv_dim_cat[ac].keys():
        inv_dim_cat[ac][coarse] = list()
    inv_dim_cat[ac][coarse].append(dim)

simple_binders = [v[1] for v in binders]
plot_dims = list()
plot_ac_coarse = list()
break_points = list()
main_break_point = list()
x_ticks = list()
lower_ticks = list()
counter = 0
for y_a in sorted(inv_dim_cat.keys(), reverse=True):
    for y_b, y_dims in sorted(inv_dim_cat[y_a].items()):
        for y_d in y_dims:
            if y_d  not in simple_binders:
                continue
            plot_dims.append(simple_binders.index(y_d))
            lower_ticks.append(y_d)
            plot_ac_coarse.append((y_a, y_b))
            counter += 1
        if len(break_points) == 0:
            x_ticks.append((counter*0.5, y_b))
        elif counter == break_points[-1][0]:
            continue
        elif len(break_points) > 0:
            x_ticks.append((break_points[-1][0]+((counter-break_points[-1][0])*0.5), y_b))
        break_points.append((counter, y_a, y_b))
    main_break_point.append(counter)
assert counter == 48

plot_mtrx = numpy.zeros(shape=(64, 48))

plot_ws = list()
plot_cats = list()

x_start = 0
for ac, cats_ws in abs_con.items():
    for cat in cats_ws:
        ws = inv_cats[cat]
        for w in ws:
            plot_cats.append((ac, cat))
            plot_ws.append(w)
            idx = de_ws.index(w)
            for y_i, y in enumerate(plot_dims):
                plot_mtrx[x_start, y_i] = binder_norms[idx, y]
            x_start += 1
fig, ax = pyplot.subplots(
                          figsize=(15, 20),
                          )
ax.imshow(plot_mtrx)
pyplot.vlines(
              x=main_break_point[0]-.5,
              ymin=-.5,
              ymax=63.5,
              linewidth=5,
              color='fuchsia',
              )
pyplot.vlines(
              x=[v[0]-.5 for v in break_points if v[0]!=main_break_point[0]],
              ymin=-.5,
              ymax=63.5,
              linewidth=3,
              color='white',
              )
pyplot.hlines(y=31.5,
              xmin=-.5,
              xmax=47.5,
              linewidth=5,
              color='fuchsia',
              )
pyplot.hlines(y=[7.5, 15.5, 23.5, 39.5, 47.5, 55.5],
              xmin=-.5,
              xmax=47.5,
              linewidth=3,
              color='white',
              )
#pyplot.tight_layout()
#ax.tick_params(top=True, bottom=False,
#                   labeltop=True, labelbottom=False)
pyplot.yticks(ticks=range(64), labels=plot_ws)
pyplot.xticks(ticks=[v[0]-.5 for v in x_ticks], labels=[v[1] for v in x_ticks], rotation=35, ha='right', fontsize=20, fontweight='bold')
secax = ax.secondary_xaxis(
                           location='top',
                           #transform=ax.transData
                           )
secax.set_xticks(ticks=range(48), labels=lower_ticks, rotation=45, ha='left')

ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
pyplot.savefig(
               os.path.join('plots', 'norms_selected', 'full_dims',
               'heatmap.jpg'),
               pad_inches=0,
)

### covariance for dimensions
cov = numpy.corrcoef(plot_mtrx.T)
fig, ax = pyplot.subplots(
                          figsize=(20, 20),
                          )
im = ax.imshow(cov, cmap='coolwarm')
#cbar = ax.figure.colorbar(im, ax=ax,)
pyplot.colorbar(
                mappable=im,
                shrink=0.8,
                #boundaries=[-1,1.]
                )
pyplot.vlines(
              x=main_break_point[0]-.5,
              ymin=-.5,
              ymax=47.5,
              linewidth=5,
              color='fuchsia',
              )
pyplot.vlines(
              x=[v[0]-.5 for v in break_points if v[0]!=main_break_point[0]],
              ymin=-.5,
              ymax=47.5,
              linewidth=3,
              color='white',
              )
pyplot.hlines(
              y=main_break_point[0]-.5,
              xmin=-.5,
              xmax=47.5,
              linewidth=5,
              color='fuchsia',
              )
pyplot.hlines(
              y=[v[0]-.5 for v in break_points if v[0]!=main_break_point[0]],
              xmin=-.5,
              xmax=47.5,
              linewidth=3,
              color='white',
              )

pyplot.xticks(ticks=range(48), labels=lower_ticks, rotation=45, ha='right')
pyplot.yticks(ticks=range(48), labels=lower_ticks)

ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
pyplot.savefig(
               os.path.join('plots', 'norms_selected', 'full_dims',
               'covariance_dims.jpg'),
               pad_inches=0,
)

### reduced heatmap

reduced = [
           'Audition', 'Taste',
           'UpperLimb',
           #'Practice',
           'Touch', 'Smell', 'Vision',
           'Arousal', 'Caused', 'Needs', 'Harm', 'Communication', 'Time',
           ]
plot_dims = [simple_binders.index(d) for d in reduced]

plot_mtrx = numpy.zeros(shape=(12, 64))

plot_ws = list()
plot_cats = list()

x_start = 0
for ac, cats_ws in abs_con.items():
    for cat in cats_ws:
        ws = inv_cats[cat]
        for w in ws:
            plot_cats.append((ac, cat))
            plot_ws.append(w)
            idx = de_ws.index(w)
            for y_i, y in enumerate(plot_dims):
                plot_mtrx[y_i, x_start] = binder_norms[idx, y]
            x_start += 1
fig, ax = pyplot.subplots(
                          figsize=(20, 10),
                          )
ax.imshow(plot_mtrx)
pyplot.hlines(
              y=[5.5],
              xmin=-.5,
              xmax=63.5,
              linewidth=5,
              color='fuchsia',
              )
pyplot.vlines(x=31.5,
              ymin=-.5,
              ymax=11.5,
              linewidth=5,
              color='fuchsia',
              )
pyplot.vlines(x=[7.5, 15.5, 23.5, 39.5, 47.5, 55.5],
              ymin=-.5,
              ymax=11.5,
              linewidth=3,
              color='white',
              )
pyplot.xticks(ticks=range(64), labels=plot_ws, rotation=45)
pyplot.yticks(ticks=range(12), labels=reduced, fontsize=15)

ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
pyplot.savefig(
               os.path.join('plots', 'norms_selected', 'reduced_dims',
               'reduced_heatmap.jpg'),
               pad_inches=0,
               )

### reduced violinplots

fig, ax = pyplot.subplots(
                          figsize=(20, 10),
                          )
ax.violinplot(
              plot_mtrx.T,
              positions=[_-.1 for _ in range(12)],
              showextrema=False,
              side='low',
              )
ax.scatter(
           numpy.array([[_+0.1+(random.randrange(-10, 10)*0.01) for __ in range(64)] for _ in range(12)]),
           plot_mtrx,
           color='lavender',
           alpha=0.5,
           linewidth=0.5,
           edgecolor='black',
           )
ax.scatter(
           [_-.1 for _ in range(12)],
           numpy.average(plot_mtrx, axis=1),
           marker='D',
           s=100,
           linewidth=2,
           edgecolor='white',
           )
pyplot.hlines(
              y=[0, 1, 2, 3, 4, 5, 6],
              xmin=-.5,
              xmax=11.5,
              linestyle='dashed',
              color='silver',
              alpha=0.5,
              )
pyplot.xticks(ticks=range(12), labels=reduced, fontsize=15, fontweight='bold')
pyplot.ylabel(
              'Average rating',
              fontsize=15,
              fontweight='bold',
              )

ax.spines[['right', 'bottom', 'top']].set_visible(False)
pyplot.tight_layout()
pyplot.savefig(
               os.path.join('plots', 'norms_selected', 'reduced_dims',
               'reduced_violinplots_dims.jpg'),
               pad_inches=0,
               )

t_sne = manifold.TSNE(
    n_components=2,
    perplexity=8,
)
reduced_tsne = t_sne.fit_transform(plot_mtrx.T)
fig, ax = pyplot.subplots(
                          figsize=(20, 20),
                          )
for cat, ws in inv_cats.items():
    color = color_mapper[cat]
    ws_idxs = [_ for _, w in enumerate(de_ws) if w in ws]
    ax.scatter(
        reduced_tsne[ws_idxs, 0],
        reduced_tsne[ws_idxs, 1],
        color=color,
        s=750,
        linewidths=2.5,
        edgecolors='white',
           )
    for idx in ws_idxs:
        ax.text(
                x=reduced_tsne[idx, 0]+0.25,
                y=reduced_tsne[idx, 1],
                s=de_ws[idx],
                fontsize=20,
                ha='left',
                )
ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
pyplot.xticks(ticks=[])
pyplot.yticks(ticks=[])
pyplot.savefig(
               os.path.join('plots', 'norms_selected', 'reduced_dims',
               'reduced_tsne.jpg'),
               )


### covariance for dimensions
cov = numpy.corrcoef(plot_mtrx)
fig, ax = pyplot.subplots(
                          figsize=(20, 20),
                          )
im = ax.imshow(cov, cmap='coolwarm')
pyplot.colorbar(
                mappable=im,
                shrink=0.8,
                #boundaries=[-1,1.]
                )
for x in range(cov.shape[0]):
    for y in range(cov.shape[1]):
        ax.text(
                x,
                y,
                ha='center',
                va='center',
                color='white',
                fontsize=20,
                fontweight='bold',
                s=round(cov[x, y], 2)
                )
pyplot.vlines(
              x=5.5,
              ymin=-.5,
              ymax=11.5,
              linewidth=5,
              color='fuchsia',
              )
pyplot.hlines(
              y=5.5,
              xmin=-.5,
              xmax=11.5,
              linewidth=5,
              color='fuchsia',
              )

pyplot.xticks(ticks=range(12), labels=reduced, rotation=45, ha='right', fontsize=20)
pyplot.yticks(ticks=range(12), labels=reduced, fontsize=20)

ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
pyplot.savefig(
               os.path.join('plots', 'norms_selected', 'reduced_dims',
               'reduced_covariance_dims.jpg'),
               pad_inches=0,
)

### plotting by abstract/concrete concepts, separately for abstract/concrete dimensions
fig, ax = pyplot.subplots(
                          figsize=(20, 20),
                          )
assert data.shape == (64, )
corr = {'abstract' : -.5, 'concrete' : .5}
for ac, cats in abs_con.items():
    assert ac in ['1', '0']
    #x = int(ac)
    x = -1 if ac=='0' else 1
    #for cat, ws in cats.items():
    for __, ac2 in enumerate(['concrete', 'abstract']):
        #dim_idxs = [_ for _, d in enumerate(binders) if dim_cat[d[1]][1]==ac2]
        if ac2 == 'concrete':
            dim_idxs = plot_dims[:6]
        else:
            dim_idxs = plot_dims[6:]
        assert len(dim_idxs) == 6
        data = numpy.nanmean(z_rts[:, :, dim_idxs], axis=(1, 2))
        ac_idxs = list()
        for cat in cats:
            ws = inv_cats[cat]
            cat_idxs = [de_ws.index(w) for w in ws]
            rand_corrs = [random.randrange(-10, 10)*0.01 for _ in range(len(cat_idxs))]
            ac_idxs.extend(cat_idxs)
            ax.scatter(
                       [x+(corr[ac2]*0.66)+rand_corrs[_] for _ in range(len(cat_idxs))],
                       data[cat_idxs],
                       color=color_mapper[cat],
                       s=200,
                       linewidth=1,
                       edgecolor='white',
                       )
            for _, w_i in enumerate(cat_idxs):
                ax.text(
                        x=x+(corr[ac2]*0.66)+rand_corrs[_],
                        y=data[w_i],
                        s=de_ws[w_i],
                        fontsize=15,
                        )
        assert len(ac_idxs) == 32
        parts = ax.violinplot(
                  data[ac_idxs],
                  positions=[x+corr[ac2]],
                  side='low' if ac2=='abstract' else 'high',
                  showextrema=False,
                  )
        col = 'darkseagreen' if ac2=='concrete' else 'pink'
        if ac2 == 'concrete':
            label = 'concrete\ndimensions'
        else:
            label = 'abstract\ndimensions'
        if ac == '1':

            ax.scatter(
                  [x+corr[ac2]],
                  numpy.nanmean(data[ac_idxs]),
                  s=750,
                  color=col,
                   linewidth=10,
                   edgecolor='white',
                   marker='D',
                   label=label,
                  )
        else:
            ax.scatter(
                  [x+corr[ac2]],
                  numpy.nanmean(data[ac_idxs]),
                  s=750,
                  color=col,
                   linewidth=10,
                   edgecolor='white',
                   marker='D',
                  )
        for pc in parts['bodies']:
            pc.set_facecolor(col)
ax.legend(fontsize=30, ncols=2, loc=9)
pyplot.hlines(
              y=0.,
              xmin=-1.75,
              xmax=1.75,
              color='black',
              alpha=0.6,
              )
pyplot.xticks(
              ticks=[-1.4, 1.4],
              labels=['Abstract\nconcepts', 'Concrete\nconcepts'],
              fontsize=40,
              fontweight='bold',
              )
pyplot.yticks(
              fontsize=30,
              )
pyplot.ylabel(
              'Z-scored RT across all 48 dimensions and subjects',
              fontsize=30,
              fontweight='bold',
              )
pyplot.tight_layout()
pyplot.savefig(
               os.path.join('plots', 'norms_selected', 'reduced_dims',
               'reduced_cats_dims_rts.jpg'),
               pad_inches=0,
               )
