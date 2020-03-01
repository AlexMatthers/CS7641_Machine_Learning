import mlrose_hiive as mlr
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt
import multiprocessing as MP
import numpy as np
import pandas as pd
import os
import sys
import pickle
import hashlib
import time
from pathlib import Path
SEED = 10
THREADS = -1
VERBOSE = False
OUTPUT_DIRECTORY = os.path.join(sys.path[0],'NNOutput')
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(os.path.join(OUTPUT_DIRECTORY, 'NN_storage')).mkdir(parents=True, exist_ok=True)

DS_1_fName = os.path.join("Epileptic Seizure Recognition", "PPData.csv")
DS_1 = pd.read_csv(os.path.join(sys.path[0], DS_1_fName)).to_numpy()[:, 1:].astype(int)
np.random.shuffle(DS_1)

ss = MinMaxScaler()
oh = OneHotEncoder()

X, y = DS_1[:, :-1], DS_1[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=SEED)
X_train_scaled = ss.fit_transform(X_train).astype(int)
X_test_scaled = ss.transform(X_test).astype(int)

y_train_hot = oh.fit_transform(y_train.reshape(-1, 1)).todense().astype(int)  # One hot encode and convert to column matrix
y_test_hot = oh.transform(y_test.reshape(-1, 1)).todense().astype(int)

# lrs = [0.002]  # 6
# lrs = [0.002, 0.004, 0.02, 0.06, 0.2, 0.8]  # 6
lrs = [0.003,  0.05, 0.7]  # 6
# mil = [100]  # 4
# mil = [100, 500, 1000, 2500]  # 4
mil = [145, 435, 1300]  # 4
# acti = ['tanh']
acti = ['relu', 'sigmoid', 'tanh']
NNparams = np.array(np.meshgrid(lrs,mil,acti),dtype=object).T.reshape(-1,3)
# rsts = [10]
rsts = [7, 12]
# rsts = [5, 10, 15]
# schs = [mlr.ArithDecay(15)]
schs = [mlr.ArithDecay(25), mlr.ExpDecay(25), mlr.GeomDecay(25)]
# schs = [mlr.ArithDecay(15), mlr.ExpDecay(15), mlr.GeomDecay(15),
        # mlr.ArithDecay(175), mlr.ExpDecay(175), mlr.GeomDecay(175)]
# pops = [225]
pops = [150, 225]
# pops = [75, 150, 225]
# muts = [0.25]
muts = [0.33, 0.66]
# muts = [0.25, 0.5, 0.75]
gaParams = np.array(np.meshgrid(pops, muts)).T.reshape(-1,2)

experiments = []
for lr, mi, act in NNparams:
    for rst in rsts:
        experiments.append(mlr.NeuralNetwork(hidden_nodes=[7,5],
                                activation=act,
                                algorithm='random_hill_climb',
                                max_iters=mi,
                                learning_rate=lr,
                                restarts=rst,
                                random_state=SEED,
                                curve=True))
    for sch in schs:
        experiments.append(mlr.NeuralNetwork(hidden_nodes=[7,5],
                                activation=act,
                                algorithm='simulated_annealing',
                                max_iters=mi,
                                learning_rate=lr,
                                schedule=sch,
                                random_state=SEED,
                                curve=True))
    for pop, mut in gaParams:
        experiments.append(mlr.NeuralNetwork(hidden_nodes=[7,5],
                                activation=act,
                                algorithm='genetic_alg',
                                max_iters=mi,
                                learning_rate=lr,
                                pop_size=pop,
                                mutation_prob=mut,
                                random_state=SEED,
                                curve=True))
    experiments.append(mlr.NeuralNetwork(hidden_nodes=[7,5],
                                activation=act,
                                algorithm='gradient_descent',
                                max_iters=mi,
                                learning_rate=lr,
                                random_state=SEED,
                                curve=True))

len_e = len(experiments)
# print(len_e)
# input('Paused....')
# if input('>>') == 'y':
#     quit()
def build_list_from_file():
    results = []
    dirTarget = os.path.join(OUTPUT_DIRECTORY, 'NN_storage')
    pFiles = os.listdir(dirTarget)
    for file in pFiles:
        print(file)
        with open(os.path.join(dirTarget, file), 'rb') as pk:
            results.append(pickle.load(pk))
    return results

def handle_results_list(results_list):
    print('Handling results...')
    best_RHC_bas = 0
    best_RHC = None
    best_SA_bas = 0
    best_SA = None
    best_GA_bas = 0
    best_GA = None
    best_GD_bas = 0
    best_GD = None
    for exp in results_list:
        y_score = oh.inverse_transform(exp.predict(X_test_scaled))
        bas = balanced_accuracy_score(y_test, y_score)
        if exp.algorithm == 'random_hill_climb':
            if bas > best_RHC_bas:
                best_RHC_bas = bas
                best_RHC = exp
            elif (bas == best_RHC_bas) and (best_RHC is not None):
                if exp.max_iters < best_RHC.max_iters:  # Less complexity?
                    best_RHC = exp
        if exp.algorithm == 'simulated_annealing':
            if bas > best_SA_bas:
                best_SA_bas = bas
                best_SA = exp
            elif (bas == best_SA_bas) and (best_SA is not None):
                if exp.max_iters < best_SA.max_iters:  # Less complexity?
                    best_SA = exp
        if exp.algorithm == 'genetic_alg':
            if bas > best_GA_bas:
                best_GA_bas = bas
                best_GA = exp
            elif (bas == best_GA_bas) and (best_GA is not None):
                if exp.max_iters < best_GA.max_iters:  # Less complexity?
                    best_GA = exp
        if exp.algorithm == 'gradient_descent':
            if bas > best_GD_bas:
                best_GD_bas = bas
                best_GD = exp
            elif (bas == best_GD_bas) and (best_GD is not None):
                if exp.max_iters < best_GD.max_iters:  # Less complexity?
                    best_GD = exp

    print('Plotting results...')

    fitFig, fitAx = plt.subplots()
    fitAx.set_title('NN Weight Fitness (Loss) per iteration')
    fitAx.set_xlabel('Iterations')
    fitAx.set_ylabel('Fitness/Loss')
    fitAx.plot(best_RHC.fitness_curve, color='red', label='RHC')
    fitAx.plot(best_SA.fitness_curve, color='blue', label='SA')
    fitAx.plot(best_GA.fitness_curve, color='green', label='GA')
    # fitAx.plot(best_GD.fitness_curve, color='orange', label='GD')
    fitAx.legend()
    fitFig.set_size_inches((10,10))
    plt.savefig(os.path.join(OUTPUT_DIRECTORY,'FCurves.png'), dpi=400, format='png')
    plt.close(fitFig)

    LCFig, LCAx = plt.subplots()
    LCAx.set_title('Learning Curves for Randomly Optimized NNs')
    LCAx.set_xlabel('Training Samples')
    LCAx.set_ylabel('Score')
    t_sizes = np.linspace(0.1, 1.0, 10)
    _, train_scores_LC, valid_scores_LC = learning_curve(best_RHC,
                                                         X_train_scaled,
                                                         y_train,
                                                         n_jobs=THREADS,
                                                         train_sizes=t_sizes)
    train_scores_mean_LC, valid_scores_mean_LC = train_scores_LC.mean(axis=1), valid_scores_LC.mean(axis=1)
    LCAx.plot(t_sizes, train_scores_mean_LC, color='red', label='RHC Training')
    LCAx.plot(t_sizes, valid_scores_mean_LC, color='darkred', label='RHC Validation')
    _, train_scores_LC, valid_scores_LC = learning_curve(best_SA,
                                                         X_train_scaled,
                                                         y_train,
                                                         n_jobs=THREADS,
                                                         train_sizes=t_sizes)
    train_scores_mean_LC, valid_scores_mean_LC = train_scores_LC.mean(axis=1), valid_scores_LC.mean(axis=1)
    LCAx.plot(t_sizes, train_scores_mean_LC, color='blue', label='SA Training')
    LCAx.plot(t_sizes, valid_scores_mean_LC, color='darkblue', label='SA Validation')
    _, train_scores_LC, valid_scores_LC = learning_curve(best_GA,
                                                         X_train_scaled,
                                                         y_train,
                                                         n_jobs=THREADS,
                                                         train_sizes=t_sizes)
    train_scores_mean_LC, valid_scores_mean_LC = train_scores_LC.mean(axis=1), valid_scores_LC.mean(axis=1)
    LCAx.plot(t_sizes, train_scores_mean_LC, color='limegreen', label='GA Training')
    LCAx.plot(t_sizes, valid_scores_mean_LC, color='darkgreen', label='GA Validation')
    _, train_scores_LC, valid_scores_LC = learning_curve(best_GD,
                                                         X_train_scaled,
                                                         y_train,
                                                         n_jobs=THREADS,
                                                         train_sizes=t_sizes)
    train_scores_mean_LC, valid_scores_mean_LC = train_scores_LC.mean(axis=1), valid_scores_LC.mean(axis=1)
    LCAx.plot(t_sizes, train_scores_mean_LC, color='orange', label='GD Training')
    LCAx.plot(t_sizes, valid_scores_mean_LC, color='darkorange', label='GD Validation')
    LCAx.legend()
    LCFig.set_size_inches((10,10))
    plt.savefig(os.path.join(OUTPUT_DIRECTORY,'LCurves.png'), dpi=400, format='png')
    plt.close(LCFig)

    y_score_RHC = oh.inverse_transform(best_RHC.predict(X_test_scaled))
    y_score_SA = oh.inverse_transform(best_SA.predict(X_test_scaled))
    y_score_GA = oh.inverse_transform(best_GA.predict(X_test_scaled))
    y_score_GD = oh.inverse_transform(best_GD.predict(X_test_scaled))
    y_proba_RHC = best_RHC.predicted_probs[:,1]
    y_proba_SA = best_SA.predicted_probs[:,1]
    y_proba_GA = best_GA.predicted_probs[:,1]
    y_proba_GD = best_GD.predicted_probs[:,1]
    fpr_RHC, tpr_RHC, _ = roc_curve(y_test, y_proba_RHC)
    auc_scr_RHC = roc_auc_score(y_test, y_proba_RHC)
    fpr_SA, tpr_SA, _ = roc_curve(y_test, y_proba_SA)
    auc_scr_SA = roc_auc_score(y_test, y_proba_SA)
    fpr_GA, tpr_GA, _ = roc_curve(y_test, y_proba_GA)
    auc_scr_GA = roc_auc_score(y_test, y_proba_GA)
    fpr_GD, tpr_GD, _ = roc_curve(y_test, y_proba_GD)
    auc_scr_GD = roc_auc_score(y_test, y_proba_GD)
    ROC_Fig, ROC_Ax = plt.subplots()
    ROC_Ax.set_title("Receiver Operating Characteristic")
    ROC_Ax.set_xlabel('False Positive Rate')
    ROC_Ax.set_ylabel('True Positive Rate')
    ROC_Ax.plot(fpr_RHC, tpr_RHC, color='red', label=f'RHC ROC Curve (area = {auc_scr_RHC})')
    ROC_Ax.plot(fpr_SA, tpr_SA, color='blue', label=f'SA ROC Curve (area = {auc_scr_SA})')
    ROC_Ax.plot(fpr_GA, tpr_GA, color='green', label=f'GA ROC Curve (area = {auc_scr_GA})')
    ROC_Ax.plot(fpr_GD, tpr_GD, color='orange', label=f'GD ROC Curve (area = {auc_scr_GD})')
    ROC_Ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ROC_Ax.legend(loc="lower right")
    ROC_Fig.set_size_inches((10,10))
    plt.savefig(os.path.join(OUTPUT_DIRECTORY,'ROC.png'), dpi=400, format='png')
    plt.close(ROC_Fig)

    with open(os.path.join(OUTPUT_DIRECTORY, 'RHC_Report.txt'), 'w') as RHC_text:
        _ = RHC_text.write(f'{confusion_matrix(y_test, y_score_RHC)}\n')
        _ = RHC_text.write(f'{classification_report(y_test, y_score_RHC)}')

    with open(os.path.join(OUTPUT_DIRECTORY, 'SA_Report.txt'), 'w') as SA_text:
        _ = SA_text.write(f'{confusion_matrix(y_test, y_score_SA)}\n')
        _ = SA_text.write(f'{classification_report(y_test, y_score_SA)}')

    with open(os.path.join(OUTPUT_DIRECTORY, 'GA_Report.txt'), 'w') as GA_text:
        _ = GA_text.write(f'{confusion_matrix(y_test, y_score_GA)}\n')
        _ = GA_text.write(f'{classification_report(y_test, y_score_GA)}')

    with open(os.path.join(OUTPUT_DIRECTORY, 'GD_Report.txt'), 'w') as GD_text:
        _ = GD_text.write(f'{confusion_matrix(y_test, y_score_GD)}\n')
        _ = GD_text.write(f'{classification_report(y_test, y_score_GD)}')


def runUtil(exp, X_train_scaled, y_train_hot):
    print(f'Running exp ID: {exp.uniqueid}')
    exp.time = time.perf_counter()
    return exp.fit(X_train_scaled, y_train_hot)

results_list = []
def cbReturn(result):
    print(f'Finished {result.uniqueid}')
    print(f'Time taken: {time.perf_counter() - result.time}')
    fName = f'NN_{result.algorithm}_{result.activation}_{result.max_iters}_{result.learning_rate}_{result.restarts}_{result.schedule}_{result.pop_size}_{result.mutation_prob}.p'
    with open(os.path.join(OUTPUT_DIRECTORY, 'NN_storage', fName), 'wb') as pk:
        pickle.dump(result, pk, protocol=pickle.HIGHEST_PROTOCOL)
    results_list.append(result)
    print(f'{len_e - len(results_list)} remaining...')

def main():
    # pool = MP.Pool(12)
    pool = MP.Pool(MP.cpu_count(), maxtasksperchild=1)
    print(f'# of experiments: {len_e} on {MP.cpu_count()} processes....')
    for exp in experiments:
        exp.uniqueid = f'{exp.algorithm}_{exp.activation}_{exp.max_iters}_{exp.learning_rate}_{exp.restarts}_{exp.schedule}_{exp.pop_size}_{exp.mutation_prob}'
        # exp.uniqueid = hashlib.md5(f'{exp.algorithm}_{exp.activation}_{exp.max_iters}_{exp.learning_rate}_{exp.restarts}_{exp.schedule}_{exp.pop_size}_{exp.mutation_prob}'.encode('utf-8')).hexdigest()
        pool.apply_async(runUtil, args=(exp, X_train_scaled, y_train_hot, ), callback=cbReturn)
    pool.close()
    pool.join()
    handle_results_list(results_list)

if __name__ == '__main__':
    if input('Do main? >') == 'y':
        main()
    else:
        handle_results_list(build_list_from_file())
