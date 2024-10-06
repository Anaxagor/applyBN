import pandas as pd
from pathlib import Path

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score
from applybn.data_generation.class_balance import BNClassBalancer

from copy import copy
from imblearn.datasets import fetch_datasets
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tqdm import tqdm


def train_models(models, x_train, y_train):
    trained_models = copy(models)

    for model in trained_models:
        model.fit(x_train, y_train)

    return trained_models


def evaluate_models(models, x_test, y_test):
    f1_test_res = []

    for model in models:
        f1_test = f1_score(y_test, model.predict(x_test))
        f1_test_res.append(f1_test)

    return f1_test_res


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

ds_names = ['ecoli',
            'abalone',
            'wine_quality',
            'yeast_me2',
            'mammography',
            'abalone_19', ]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    XGBClassifier(n_estimators=50)
]

le = LabelEncoder()
class_balancer = BNClassBalancer()

data_dir = Path("/data")

res = {'model': [],
       'data_name': [],

       'f1_bn': [],
       'f1_smote': []
       }

for ds_name in tqdm(ds_names):
    ds = fetch_datasets()[ds_name]
    x, y = ds['data'], ds['target']
    df = pd.DataFrame(x)
    df['class'] = le.fit_transform(y)

    disbalanced_data = df.sample(frac=1).reset_index(drop=True)
    size = len(disbalanced_data)
    train, test = disbalanced_data.iloc[:int(0.7 * size)], disbalanced_data.iloc[int(0.7 * size):]

    class_balancer.fit(train)
    bn_balanced = class_balancer.balance(train)
    bn_x, bn_y = bn_balanced[bn_balanced.columns[:-1]], bn_balanced[bn_balanced.columns[-1]]

    train_x, train_y = train[train.columns[:-1]], train[train.columns[-1]]
    test_x, test_y = test[test.columns[:-1]], test[test.columns[-1]]

    sm = SMOTE(random_state=42)
    smote_x, smote_y = sm.fit_resample(disbalanced_data[disbalanced_data.columns[:-1]],
                                   disbalanced_data[disbalanced_data.columns[-1]])

    models_bn = train_models(classifiers, bn_x, bn_y)
    f1_bn = evaluate_models(models_bn, test_x, test_y)

    models_smote = train_models(classifiers, smote_x, smote_y)
    f1_smote = evaluate_models(models_smote, test_x, test_y)

    res['model'] += model_names
    res['data_name'] += [ds_name] * len(model_names)
    res['f1_smote'] += f1_smote
    res['f1_bn'] += f1_bn

    pd.DataFrame(res).to_csv('results/exp2/results.csv')
