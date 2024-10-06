import numpy as np
from pandas import read_csv, Series, DataFrame
from pathlib import Path
from xgboost import XGBClassifier
from applybn.data_generation.get_bn import get_hybrid_bn
from sklearn.preprocessing import KBinsDiscretizer
from scipy.spatial.distance import jensenshannon


def js_div_columns(a1, a2, k=10):
    discretizer = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')
    a1_d = discretizer.fit_transform(a1.values.reshape((len(a1), 1)))
    a2_d = discretizer.transform(a2.values.reshape((len(a2), 1)))

    a1_d = np.vstack((a1_d, np.linspace(0, k-1, k).reshape((k, 1))))
    a2_d = np.vstack((a2_d, np.linspace(0, k-1, k).reshape((k, 1))))

    a1_p = Series(a1_d.reshape(len(a1) + k)).value_counts() / len(a1)
    a2_p = Series(a2_d.reshape(len(a2) + k)).value_counts() / len(a2)

    return jensenshannon(a1_p, a2_p)


def js_div_dataframes(d1, d2):
    res = np.zeros(len(d1.columns))

    for i, col in enumerate(d1.columns):
        res[i] = js_div_columns(d1[col], d2[col])

    return res.mean()


data_dir = Path('data')
clf = XGBClassifier(n_estimators=50)
res = {'ds_name': [],
       'js_divergence': []}


for data_path in data_dir.iterdir():
    print(data_path.name)

    if data_path.is_dir():
        continue

    real_data = read_csv(data_path, index_col=0).sample(1000)
    bn = get_hybrid_bn(real_data)
    synth_data = bn.sample(1000).astype(real_data.dtypes.to_dict())

    res['ds_name'].append(data_path.name[:-4])
    res['js_divergence'].append(js_div_dataframes(real_data, synth_data))

    DataFrame(res).to_csv('results/results.csv')
