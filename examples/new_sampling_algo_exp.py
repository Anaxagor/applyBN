from pandas import read_csv, concat
from pathlib import Path
import pandas as pd

from pandas.core.interchange.dataframe_protocol import DataFrame
from xgboost import XGBClassifier
from get_bn import get_hybrid_bn
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from seaborn import kdeplot
from matplotlib import pyplot as plt

# Получение результатов

# data_dir = Path('/home/ilia/PycharmProjects/applyBN_old/data')
# clf_standard = XGBClassifier(n_estimators=50)
# clf_synth = XGBClassifier(n_estimators=50)
#
# ds_names = ['MagicTelescope', 'bank-marketing', 'california', 'credit']
#
# res = {'ds_name': [],
#        'standard_f1_sample': [],
#        'standard_f1_pop': [],
#        'standard_f1_rel': [],
#
#        'synth_f1_sample': [],
#        'synth_f1_pop': [],
#        'synth_f1_rel': [],
#        }
#
# for ds_name in ds_names:
#     for i in tqdm(range(50)):
#         full_data = read_csv(f'data/{ds_name}.csv', index_col=0)
#         full_x, full_y = full_data[full_data.columns[:-1]], full_data[full_data.columns[-1]]
#
#         synth_test = read_csv(f'res_data/new_sampling_reject/{ds_name}.csv', index_col=0)[full_data.columns]
#         synth_x, synth_y = synth_test[synth_test.columns[:-1]], synth_test[synth_test.columns[-1]]
#
#         sample = full_data.sample(2000)
#
#         size = len(sample)
#         sample_x, sample_y = sample[sample.columns[:-1]], sample[sample.columns[-1]]
#         sample_x_train, sample_y_train = sample_x.iloc[:int(0.7 * size)], sample_y.iloc[:int(0.7 * size)]
#         sample_x_test, sample_y_test = sample_x.iloc[int(0.7 * size):], sample_y.iloc[int(0.7 * size):]
#
#         clf_standard.fit(sample_x_train, sample_y_train)
#         standard_f1_sample = f1_score(sample_y_test, clf_standard.predict(sample_x_test))
#         standard_f1_pop = f1_score(full_y, clf_standard.predict(full_x))
#         standard_f1_rel = abs(standard_f1_pop - standard_f1_sample) / standard_f1_pop
#
#         clf_synth.fit(sample_x, sample_y)
#         synth_f1_sample = f1_score(synth_y, clf_synth.predict(synth_x))
#         synth_f1_pop = f1_score(full_y, clf_synth.predict(full_x))
#         synth_f1_rel = abs(synth_f1_pop - synth_f1_sample) / synth_f1_pop
#
#         res['ds_name'].append(ds_name)
#
#         res['standard_f1_sample'].append(standard_f1_sample)
#         res['standard_f1_pop'].append(standard_f1_pop)
#         res['standard_f1_rel'].append(standard_f1_rel)
#
#         res['synth_f1_sample'].append(synth_f1_sample)
#         res['synth_f1_pop'].append(synth_f1_pop)
#         res['synth_f1_rel'].append(synth_f1_rel)
#
#         pd.DataFrame(res).to_csv('new_sampling_res_reject.csv')

# Рисование

ds_names = ['MagicTelescope', 'bank-marketing', 'california', 'credit']
res = read_csv('new_sampling_res_reject.csv', index_col=0)

# synth_f1_sample = res['synth_f1_pop']
# synth_f1_pop = res['synth_f1_sample']
# synth_f1_rel = res['synth_f1_rel']

# plt.scatter(synth_f1_sample, synth_f1_rel)
# # plt.plot([0.5, 1.0], [0.5, 1.0], linestyle=':')
# plt.show()

for ds_name in ds_names:
    cur_res = res[res['ds_name'] == ds_name]
    standard_f1_sample = cur_res['standard_f1_sample']
    standard_f1_pop = cur_res['standard_f1_pop']
    standard_f1_rel = cur_res['standard_f1_rel']

    synth_f1_sample = cur_res['synth_f1_sample']
    synth_f1_pop = cur_res['synth_f1_pop']
    synth_f1_rel = cur_res['synth_f1_rel']

    # kdeplot(synth_f1_pop, label='Synthetic pop')
    kdeplot(synth_f1_sample, label='Synthetic sample')
    kdeplot(standard_f1_sample, label='Standard sample')
    plt.legend()
    plt.show()
