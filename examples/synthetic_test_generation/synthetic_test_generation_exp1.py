import numpy as np
from pandas import read_csv
from pathlib import Path

from seaborn import kdeplot
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from matplotlib import pyplot as plt
from applybn.data_generation.synthetic_test_generation import BNTestGenerator


data_dir = Path('data')

real_data_lens = [500, 1000, 2000]
clfs = [('xgb', XGBClassifier(n_estimators=50)),
        ('knn', KNeighborsClassifier(3)),
        ('mlp', MLPClassifier(alpha=1, max_iter=1000, random_state=42)),]

for data_path in data_dir.iterdir():
    for real_data_len in real_data_lens:
        full_data = read_csv(data_path, index_col=0)
        real_data = full_data.sample(real_data_len)
        test_generator = BNTestGenerator()
        test_generator.fit(real_data)

        for clf_name, clf in clfs:
            if data_path.is_dir():
                continue

            reject_synth_data = test_generator.generate_test_data(real_data, clf)
            vanilla_synth_data = test_generator.bn.sample(len(reject_synth_data)).astype(real_data.dtypes.to_dict())[full_data.columns]

            real_f1 = test_generator.evaluate_data(real_data, clf, f1_score, 100)
            reject_synth_f1 = test_generator.evaluate_data(reject_synth_data, clf, f1_score, 100)
            vanilla_synth_f1 = test_generator.evaluate_data(vanilla_synth_data, clf, f1_score,100)

            kdeplot(real_f1, label='Real sample')
            kdeplot(vanilla_synth_f1, label='Vanilla synthetic sample')
            kdeplot(reject_synth_f1, label='Reject synthetic sample')
            plt.title(f'Dataset : {data_path.name[:-4]}')
            plt.legend()
            plt.savefig(f'synthetic_test_generation_results/{data_path.name[:-4]}_{real_data_len}_{clf_name}.png')
            plt.clf()

            reject_synth_data.to_csv(f'results/datasets/exp1/'
                                     f'synth_{data_path.name[:-4]}_{real_data_len}_{clf_name}.csv')
            real_data.to_csv(f'results/datasets/exp1/'
                                     f'real_{data_path.name[:-4]}_{real_data_len}_{clf_name}.csv')

