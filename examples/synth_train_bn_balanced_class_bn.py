import pandas as pd
from pathlib import Path

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from get_bn import get_hybrid_bn
from copy import copy

DEBUG = False


def train_models(models, pd_data_train):
    trained_models = copy(models)

    x_train, y_train = (pd_data_train[pd_data_train.columns[:-1]], pd_data_train[pd_data_train.columns[-1]])

    for model in trained_models:
        model.fit(x_train, y_train)

    return trained_models


def evaluate_models(models, pd_data_test):
    x_test, y_test = (pd_data_test[pd_data_test.columns[:-1]], pd_data_test[pd_data_test.columns[-1]])

    f1_test_res = []

    for model in models:
        f1_test = f1_score(y_test, model.predict(x_test))
        f1_test_res.append(f1_test)

    return f1_test_res


if DEBUG:
    model_names = [
        "Neural Net",
        "XGB",
    ]

    data_names = [data.name[:-4] for data in Path("data").iterdir()]

    sample_lens = [200, 500, 1000, 3000]

    initial_disbalance = [0.1, 0.5, 1.0]

    classifiers = [
        MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        XGBClassifier(n_estimators=50),
    ]
else:
    model_names = [
        "Nearest Neighbors",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
        "XGB",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        AdaBoostClassifier(algorithm="SAMME", random_state=42),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        XGBClassifier(n_estimators=50)
    ]

    sample_lens = [100, 200, 300, 400, 500, 1000, 2000, 3000]

    initial_disbalance = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

le = LabelEncoder()

data_dir = Path("data")
res = {'model': [],
       'data_name': [],
       'sample_len': [],
       'class_disbalance': [],

       'f1_test_raw_disbalanced': [],
       'f1_test_raw_population': [],

       'f1_test_bn_mix_on_disbalanced': [],
       'f1_test_bn_mix_on_population': [],
       }

for data in data_dir.iterdir():
    try:
        if data.is_dir():
            continue

        print(data.name[:-4])
        pd_data = pd.read_csv(data, index_col=0).sample(frac=1).reset_index(drop=True)
        pd_data[pd_data.columns[-1]] = le.fit_transform(pd_data[pd_data.columns[-1]])

        pd_data_true_gen, pd_data_exp = pd_data.iloc[:len(pd_data) // 2], pd_data.iloc[len(pd_data) // 2:]

        for sample_len in sample_lens:
            for disbalance in initial_disbalance:
                print(sample_len, disbalance)

                class0_len = sample_len // 2
                class1_len = int(class0_len * disbalance)

                test_sample_0 = pd_data_exp[pd_data_exp[pd_data_exp.columns[-1]] == 0].sample(class0_len)
                test_sample_1 = pd_data_exp[pd_data_exp[pd_data_exp.columns[-1]] == 1].sample(class1_len)

                disbalanced_data = pd.concat([test_sample_0, test_sample_1]).sample(frac=1).reset_index(drop=True)

                bn_mix = get_hybrid_bn(disbalanced_data, mixtures=True)
                class_len_diff = class0_len - class1_len

                if class_len_diff > 0:
                    class1 = bn_mix.sample(class_len_diff, evidence={disbalanced_data.columns[-1]: 1})[
                        disbalanced_data.columns]
                    synth_balanced_bn_mix = pd.concat([disbalanced_data, class1]).sample(frac=1).reset_index(drop=True)
                    synth_balanced_bn_mix = synth_balanced_bn_mix.astype(disbalanced_data.dtypes.to_dict())
                else:
                    synth_balanced_bn_mix = disbalanced_data

                # standard learning pipeline with synthetically balanced data
                split_i = int(0.8 * len(disbalanced_data))
                modeled_train_sample, modeled_test_sample = disbalanced_data.iloc[:split_i], disbalanced_data.iloc[split_i:]

                models_disbalanced = train_models(classifiers, modeled_train_sample)
                f1_test_raw_disbalanced = evaluate_models(models_disbalanced, modeled_test_sample)
                f1_test_raw_population = evaluate_models(models_disbalanced, pd_data_true_gen)

                models_bn_mix_balanced = train_models(classifiers, synth_balanced_bn_mix)
                f1_test_bn_mix_on_disbalanced = evaluate_models(models_bn_mix_balanced, disbalanced_data)
                f1_test_bn_mix_on_population = evaluate_models(models_bn_mix_balanced, pd_data_true_gen)

                res['model'] += model_names
                res['data_name'] += [data.name[:-4]] * len(model_names)
                res['sample_len'] += [sample_len] * len(model_names)
                res['class_disbalance'] += [disbalance] * len(model_names)

                res['f1_test_raw_disbalanced'] += f1_test_raw_disbalanced
                res['f1_test_raw_population'] += f1_test_raw_population

                res['f1_test_bn_mix_on_disbalanced'] += f1_test_bn_mix_on_disbalanced
                res['f1_test_bn_mix_on_population'] += f1_test_bn_mix_on_population

                pd.DataFrame(res).to_csv('synth_train_bn_balanced_classes_real_plus_synth.csv')
    except ValueError:
        print("Something wrong")
        continue
