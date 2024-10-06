import numpy as np
from pandas import read_csv, DataFrame
from pathlib import Path

from seaborn import kdeplot
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm

from matplotlib import pyplot as plt
from applybn.data_generation.synthetic_test_generation import BNTestGenerator

data_dir = Path('data')
test_evaluator = BNTestGenerator()

real_data_lens = [500, 1000, 2000]
clfs = [('xgb', XGBClassifier(n_estimators=50)),
        ('knn', KNeighborsClassifier(3)),
        ('mlp', MLPClassifier(alpha=1, max_iter=1000, random_state=42)),]

res = {'ds_name': [],
       'data_len': [],
       'clf_name': [],
       'f1_real': [],
       'f1_synth': [],}

total_counts = 0
match_counts = 0

for data_path in data_dir.iterdir():
    for real_data_len in real_data_lens:
        real_res = {}
        synth_res = {}
        train = read_csv(data_path, index_col=0)

        for clf_name, clf in clfs:
            clf.fit(train[train.columns[:-1]], train[train.columns[-1]])

            real_test = read_csv(f'results/datasets_exp1/'
                                     f'real_{data_path.name[:-4]}_{real_data_len}_{clf_name}.csv', index_col=0)
            real_x, real_y = real_test[real_test.columns[:-1]], real_test[real_test.columns[-1]]

            synth_test = read_csv(f'results/datasets_exp1/'
                                     f'synth_{data_path.name[:-4]}_{real_data_len}_{clf_name}.csv', index_col=0)
            synth_x, synth_y = synth_test[synth_test.columns[:-1]], synth_test[synth_test.columns[-1]]

            f1_real = f1_score(real_y, clf.predict(real_x))
            f1_synth = f1_score(real_y, clf.predict(real_x))

            res['ds_name'].append(data_path.name[:-4])
            res['data_len'].append(real_data_len)
            res['clf_name'].append(clf_name)
            res['f1_real'].append(f1_real)
            res['f1_synth'].append(f1_synth)

        rang_list_real = sorted(real_res.items(), key=lambda x: x[1])
        rang_list_synth = sorted(synth_res.items(), key=lambda x: x[1])

        total_counts += 1
        match_counts += rang_list_real == rang_list_synth


print('Ranging model lists match is ', 100 * match_counts / total_counts, ' %')
DataFrame(res).to_csv('results/results_exp2.csv')




