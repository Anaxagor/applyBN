from matplotlib import pyplot as plt
from seaborn import lineplot
from pandas import read_csv


res_data_bn = read_csv('bn_balancer.csv', index_col=0)
res_data_ctgan = read_csv('ctgan_balancer.csv', index_col=0)
res_data_smote = read_csv('smote_balancer.csv', index_col=0)

disbalance = 0.2

for disbalance in [0.2, 0.4, 0.6, 0.8, 1.0]:
    for data_name in ['bank-marketing', 'california', 'credit']:
        for model in ['XGB', 'Nearest Neighbors', 'Neural Net', 'Naive Bayes']:
            res_data_bn = read_csv('res_data/synth_train_bn_balanced_classes_new.csv', index_col=0)
            res_data_ctgan = read_csv('res_data/ctgan_balancer.csv', index_col=0)
            res_data_smote = read_csv('res_data/smote_balancer.csv', index_col=0)

            res_data_bn = res_data_bn[(res_data_bn['class_disbalance'] == disbalance) & (res_data_bn['model'] == model)
                                      & (res_data_bn['data_name'] == data_name)]
            res_data_ctgan = res_data_ctgan[(res_data_ctgan['class_disbalance'] == disbalance) & (res_data_ctgan['model'] == model)
                                            & (res_data_ctgan['data_name'] == data_name)]
            res_data_smote = res_data_smote[(res_data_smote['class_disbalance'] == disbalance) & (res_data_smote['model'] == model)
                                            & (res_data_smote['data_name'] == data_name)]

            f1_trained_raw_tested_population = res_data_bn['f1_test_raw_disbalanced']
            f1_trained_mix_bn_tested_population = res_data_bn['f1_test_bn_mix_on_population']
            f1_trained_ctgan_tested_population = res_data_ctgan['f1_test_ctgan_on_population']
            f1_trained_smote_tested_population = res_data_smote['f1_test_smote_on_population']

            x = res_data_bn['sample_len']

            print('RAW')
            print(f1_trained_raw_tested_population)

            print('BN MIX')
            print(f1_trained_mix_bn_tested_population)

            print('CTGAN')
            print(f1_trained_ctgan_tested_population)

            print('SMOTE')
            print(f1_trained_smote_tested_population)

            plt.plot(x, f1_trained_raw_tested_population, label='Disbalanced train')
            plt.plot(x, f1_trained_mix_bn_tested_population, label='Mix BN')
            plt.plot(x, f1_trained_ctgan_tested_population, label='CTGAN')
            plt.plot(x, f1_trained_smote_tested_population, label='SMOTE')

            plt.title(f'Model: {model}, disbalance: {int(disbalance*10)}/10')
            plt.legend()
            plt.xlabel('Dataset size')
            plt.ylabel('F1 score')
            plt.savefig(f'{model}_{data_name}_disbalance{disbalance}_real_plus_synth.png')
            plt.clf()
