from pandas import read_csv, concat
from pathlib import Path
from xgboost import XGBClassifier
from get_bn import get_hybrid_bn
from sklearn.metrics import f1_score
from tqdm import tqdm


def evaluate_data(data):
    clf = XGBClassifier(n_estimators=50)
    x, y = data[data.columns[:-1]], data[data.columns[-1]]

    size = len(data)
    x_train, y_train = x.iloc[:int(0.7 * size)], y.iloc[:int(0.7 * size)]
    x_test, y_test = x.iloc[int(0.7 * size):], y.iloc[int(0.7 * size):]
    clf.fit(x_train, y_train)

    return f1_score(y_test, clf.predict(x_test))

data_dir = Path('/home/ilia/PycharmProjects/applyBN_old/data')
clf = XGBClassifier(n_estimators=50)

res = {'ds_name': [],
       'real_sample_len': [],}


synth_f1 = []
real_f1 = []

for data_path in data_dir.iterdir():
    print(data_path.name)

    if data_path.is_dir():
        continue

    full_data = read_csv(data_path, index_col=0)
    bn_data = full_data.sample(2000)
    bn = get_hybrid_bn(bn_data)

    synth_test = []

    for i in tqdm(range(20)):
        cur_data = full_data.sample(1000)
        cur_x, cur_y = cur_data[cur_data.columns[:-1]], cur_data[cur_data.columns[-1]]

        size = len(cur_data)
        real_x_train, real_y_train = cur_x.iloc[:int(0.7 * size)], cur_y.iloc[:int(0.7 * size)]
        real_x_test, real_y_test = cur_x.iloc[int(0.7 * size):], cur_y.iloc[int(0.7 * size):]
        clf.fit(real_x_train, real_y_train)

        cur_f1_real = f1_score(real_y_test, clf.predict(real_x_test))
        print(f'Current F1 on real data : {cur_f1_real}')

        cur_synth_test = bn.sample(1000)
        cur_synth_test = cur_synth_test.astype(cur_data.dtypes.to_dict())
        cur_f1_synth = f1_score(cur_synth_test[cur_data.columns[-1]], clf.predict(cur_synth_test[cur_data.columns[:-1]]))
        print('On synth data:')
        print(abs(cur_f1_synth - cur_f1_real))

        if abs(cur_f1_synth - cur_f1_real) < 0.03:
            synth_test.append(cur_synth_test)
            res_synth = concat(synth_test)
            res_synth.to_csv(f'new_sampling_synth_test_abs/{data_path.name[:-4]}_synth_test_3.csv')
            # break

    # if synth_test:
    #     res_synth = concat(synth_test)
    #     res_synth.to_csv(f'new_sampling_synth_test/{data_path.name[:-4]}_synth_test_10.csv')


