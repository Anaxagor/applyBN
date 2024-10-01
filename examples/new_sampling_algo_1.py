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

    return f1_score(y_test, clf.predict(x_test))

data_dir = Path('/home/ilia/PycharmProjects/applyBN_old/data')
# clf = XGBClassifier(n_estimators=50)

res = {'ds_name': [],
       'real_sample_len': [],}


n_sampling = 1000

# for data_path in data_dir.iterdir():
#     print(data_path.name)
#     real_f1 = []
#     vanilla_synth_f1 = []
#     reject_synth_f1 = []
#     synth_test_data = []
#
#     if data_path.is_dir():
#         continue
#
#     full_data = read_csv(data_path, index_col=0)
#     bn_data = full_data.sample(2*n_sampling)
#     bn = get_hybrid_bn(bn_data)
#
#     for i in tqdm(range(10)):
#         cur_real_data = full_data.sample(n_sampling)[full_data.columns]
#         real_f1.append(evaluate_data(cur_real_data))
#
#     real_f1 = np.array(real_f1)
#     real_f1_pdf = lambda x: norm.pdf(x, loc=real_f1.mean(), scale=real_f1.std())
#     real_f1_pdf_max = real_f1_pdf(real_f1.mean())
#
#     for i in tqdm(range(100)):
#         cur_synth_data = bn.sample(n_sampling)[full_data.columns]
#         cur_synth_data = cur_synth_data.astype(cur_real_data.dtypes.to_dict())
#
#         cur_synth_f1 = evaluate_data(cur_synth_data)
#         cur_f1_pdf = real_f1_pdf(cur_synth_f1)
#         vanilla_synth_f1.append(cur_f1_pdf)
#
#         if np.random.uniform(0, high=real_f1_pdf_max) < cur_f1_pdf:
#             reject_synth_f1.append(cur_synth_f1)
#             synth_test_data.append(cur_synth_data)
#             print('Accepted')
#
#     print(len(synth_test_data))
#     common_synth_data = pd.concat(synth_test_data).reset_index(drop=True)
#     print(len(common_synth_data))
#     common_synth_data.to_csv(f'res_data/new_sampling/{data_path.name[:-4]}.csv')
#     print(reject_synth_f1)
#
#
#     kdeplot(real_f1, label='Real sample')
#     kdeplot(vanilla_synth_f1, label='Vanilla synthetic sample')
#     kdeplot(reject_synth_f1, label='Reject synthetic sample')
#     plt.title(f'Dataset : {data_path.name[:-4]}')
#     plt.legend()
#     plt.savefig(f'res_plot/new_sampling/{data_path.name[:-4]}.png')
#     plt.clf()

from seaborn import pairplot


ds_names = ['bank-marketing', 'MagicTelescope', 'california', 'credit']

for ds_name in ds_names:
    real_data = read_csv(f'data/{ds_name}.csv', index_col=0).sample(100)
    synth_data = read_csv(f'res_data/new_sampling_reject/{ds_name}.csv', index_col=0).sample(100)[real_data.columns]

    real_data['domain'] = ['real' for _ in range(100)]
    synth_data['domain'] = ['synth' for _ in range(100)]

    # print(real_data)
    # print(synth_data)

    pairplot(concat([real_data, synth_data]), hue='domain')
    plt.show()

    break
