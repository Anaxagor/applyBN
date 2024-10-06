from matplotlib import pyplot as plt
from pandas import read_csv
import numpy as np

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

res = read_csv('results.csv', index_col=0)

for ds_name in ds_names:
    cur_res = res[res['data_name'] == ds_name]

    f1_bn = cur_res['f1_bn']
    f1_smote = cur_res['f1_smote']

    x = np.linspace(0, 10, len(f1_bn))
    plt.bar(x, f1_bn, width=0.5, label='F1 BN')
    plt.bar(x + 0.5, f1_smote, width=0.5, label='F1 SMOTE')
    plt.xticks(x, model_names, rotation=75)
    plt.legend()
    plt.title(f'Dataset : {ds_name}')
    plt.tight_layout()
    plt.savefig(f'by_datasets/{ds_name}.png')
    plt.clf()


for model in model_names:
    cur_res = res[res['model'] == model]

    f1_bn = cur_res['f1_bn']
    f1_smote = cur_res['f1_smote']

    x = np.linspace(0, 10, len(f1_bn))
    plt.bar(x, f1_bn, width=0.5, label='F1 BN')
    plt.bar(x + 0.5, f1_smote, width=0.5, label='F1 SMOTE')
    plt.xticks(x, cur_res['data_name'], rotation=75)
    plt.legend()
    plt.title(f'Model : {model}')
    plt.tight_layout()
    plt.savefig(f'by_models/{model}.png')
    plt.clf()
