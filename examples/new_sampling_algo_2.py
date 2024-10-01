import numpy as np
import pandas as pd
from idna.idnadata import scripts
from pandas import read_csv, concat
from pathlib import Path

from seaborn import kdeplot
from xgboost import XGBClassifier
from get_bn import get_hybrid_bn
from sklearn.metrics import f1_score
from tqdm import tqdm

from matplotlib import pyplot as plt
from scipy.stats import norm


def evaluate_data(data):
    clf = XGBClassifier(n_estimators=50)
    x, y = data[data.columns[:-1]], data[data.columns[-1]]

    size = len(data)
    x_train, y_train = x.iloc[:int(0.7 * size)], y.iloc[:int(0.7 * size)]
    x_test, y_test = x.iloc[int(0.7 * size):], y.iloc[int(0.7 * size):]
    clf.fit(x_train, y_train)

    return clf, f1_score(y_test, clf.predict(x_test))


def f1_model_with_data(model, data):
    x, y = data[data.columns[:-1]], data[data.columns[-1]]
    return f1_score(y, model.predict(x))


def train_model(model, data):
    x, y = data[data.columns[:-1]], data[data.columns[-1]]
    model.fit(x, y)

    return model


data_dir = Path('data')
full_clf = XGBClassifier(n_estimators=50)

res_synth = {'ds_name': [],
             'n_sampling': [],
             'synth_f1': []}

res_real = {'ds_name': [],
            'n_sampling': [],
            'real_f1': []}


n_sampling = 100

for n_sampling in [100, 200, 500, 1000, 2000]:
    for data_path in data_dir.iterdir():
        print(data_path.name)
        vanilla_synth_f1 = []
        reject_synth_f1 = []
        synth_test_data = []

        if data_path.is_dir():
            continue

        full_data = read_csv(data_path, index_col=0)
        synth_f1_lst = []
        real_f1_lst = []

        for k in range(10):
            real_train = full_data.sample(n_sampling)[full_data.columns]

            _, train_f1 = evaluate_data(real_train)

            bn = get_hybrid_bn(real_train)

            for i in tqdm(range(20)):
                synth_sample = bn.sample(n_sampling)[full_data.columns]
                synth_sample = synth_sample.astype(real_train.dtypes.to_dict())

                _, synth_f1 = evaluate_data(synth_sample)

                if abs(synth_f1 - train_f1) / train_f1 < 0.05:
                    synth_test_data.append(synth_sample)
                    print('Accepted')

            if synth_test_data:
                synth_test = pd.concat(synth_test_data).reset_index(drop=True)

                full_clf = train_model(full_clf, real_train)
                synth_f1 = f1_model_with_data(full_clf, synth_test)

                res_synth['ds_name'].append(data_path.name[:-4])
                res_synth['n_sampling'].append(n_sampling)
                res_synth['synth_f1'].append(synth_f1)

                pd.DataFrame(res_synth).to_csv('res_data/new_sampling_synth.csv')

                synth_f1_lst.append(synth_f1)

        if synth_test_data:
            for i in range(100):
                real_test = full_data.sample(n_sampling)[full_data.columns]
                real_f1 = f1_model_with_data(full_clf, real_test)
                real_f1_lst.append(real_f1)

                res_real['ds_name'].append(data_path.name[:-4])
                res_real['n_sampling'].append(n_sampling)
                res_real['real_f1'].append(real_f1)

                pd.DataFrame(res_real).to_csv('res_data/new_sampling_real.csv')

            kdeplot(synth_f1_lst, label='Synthetic test')
            kdeplot(real_f1_lst, label='Real test')
            plt.title(f'Dataset : {data_path.name[:-4]}, sample size : {n_sampling}')
            plt.legend()
            plt.savefig(f'res_plot/new_sampling_algo/003/{data_path.name[:-4]}_{n_sampling}.png')
            plt.clf()




#
#
#     kdeplot(real_f1, label='Real sample')
#     kdeplot(vanilla_synth_f1, label='Vanilla synthetic sample')
#     kdeplot(reject_synth_f1, label='Reject synthetic sample')
#     plt.title(f'Dataset : {data_path.name[:-4]}')
#     plt.legend()
#     plt.savefig(f'res_plot/new_sampling/{data_path.name[:-4]}.png')
#     plt.clf()
#
# from seaborn import pairplot
#
#
# ds_names = ['bank-marketing', 'MagicTelescope', 'california', 'credit']
#
# for ds_name in ds_names:
#     real_data = read_csv(f'data/{ds_name}.csv', index_col=0).sample(100)
#     synth_data = read_csv(f'res_data/new_sampling_reject/{ds_name}.csv', index_col=0).sample(100)[real_data.columns]
#
#     real_data['domain'] = ['real' for _ in range(100)]
#     synth_data['domain'] = ['synth' for _ in range(100)]
#
#     # print(real_data)
#     # print(synth_data)
#
#     pairplot(concat([real_data, synth_data]), hue='domain')
#     plt.show()
#
#     break
