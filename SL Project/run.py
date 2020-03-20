from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
import os
import sys
from pathlib import Path
SEED = 10
THREADS = 12
OUTFILE = r"Results"
VERBOSE = 0
np.random.seed(SEED)

# Data import
DS_1_fName = r"Data\Epileptic Seizure Recognition\PPData.csv"
DS_1 = pd.read_csv(os.path.join(sys.path[0], DS_1_fName)).to_numpy()[:, 1:].astype(int)
DS_1_X = DS_1[:, :-1]
DS_1_Y = DS_1[:, -1]

DS_2_fName = r"Data\MAGIC Gamma Telescope\PPData.csv"
DS_2 = pd.read_csv(os.path.join(sys.path[0], DS_2_fName)).to_numpy()
DS_2_X = DS_2[:, :-1]
DS_2_Y = DS_2[:, -1]


DSNAMES = ['Epileptic Seizure Recognition', 'MAGIC Gamma Telescope']
np.random.shuffle(DS_1)
np.random.shuffle(DS_2)

DS = [DS_1, DS_2]

algos = ['DT', 'SVM', 'kNN', 'ANN', 'BDT']

for ds_i, ds in enumerate(DS):
    X, y = ds[:, :-1], ds[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=SEED)
    # print(np.unique(y_train, return_counts=True))
    # quit()
    Path(os.path.join(OUTFILE, DSNAMES[ds_i])).mkdir(parents=True, exist_ok=True)

    if 'DT' in algos:
        print()
        print('Running Decision Tree...')
        print('--- --- --- --- --- --- ---')
        textOut = open(os.path.join(OUTFILE, DSNAMES[ds_i], 'DTresults.txt'), 'w')
        param_grid = {'criterion': ['gini', 'entropy'],
                      'max_depth': np.arange(1, 51)}
        mlp = make_pipeline(StandardScaler(), GridSearchCV(DecisionTreeClassifier(random_state=SEED), scoring='f1', param_grid=param_grid, n_jobs=THREADS, refit=True, verbose=VERBOSE))
        mlp.fit(X_train, y_train)
        ss = mlp['standardscaler']
        grid = mlp['gridsearchcv']
        bestEst = grid.best_estimator_
        bestParams = grid.best_params_
        ccpAlphas = bestEst.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
        print('Best DT params:')
        textOut.write(f'Best DT params:\n{grid.best_params_}\n')
        print(grid.best_params_)
        print('--- --- --- --- --- --- ---')

        fig, axes = plt.subplots(1,3)
        fig.suptitle('Decision Tree Graphs')
        # Build Learning Curve
        train_sizes_LC, train_scores_LC, valid_scores_LC = learning_curve(
            bestEst, X_train, y_train, n_jobs=THREADS, train_sizes=np.linspace(0.1, 1.0, 10))
        axes[0].set_title('Learning Curve')
        axes[0].set_xlabel("Training Samples")
        axes[0].set_ylabel("Score")
        train_scores_mean_LC, valid_scores_mean_LC = train_scores_LC.mean(axis=1), valid_scores_LC.mean(axis=1)
        axes[0].plot(train_sizes_LC, train_scores_mean_LC, 'o-', color='red', label="Training Score")
        axes[0].plot(train_sizes_LC, valid_scores_mean_LC, 'o-', color='green', label="Cross-validation Score")
        axes[0].legend()

        # Build Validation Curve
        param_range = np.logspace(np.log10(ccpAlphas[1])-1, np.log10(ccpAlphas[-1])+1,15)
        param_name = "ccp_alpha"
        train_scores_VC, valid_scores_VC = validation_curve(
            bestEst, X_train, y_train, param_name=param_name, param_range=param_range, scoring="f1", n_jobs=THREADS)
        axes[1].set_title('Complexity/Validation Curve')
        axes[1].set_xlabel(r"$\alpha$")
        axes[1].set_ylabel("Score")
        train_scores_mean_VC, valid_scores_mean_VC = train_scores_VC.mean(axis=1), valid_scores_VC.mean(axis=1)
        axes[1].semilogx(param_range, train_scores_mean_VC, 'o-', color='red', label="Training Score")
        axes[1].semilogx(param_range, valid_scores_mean_VC, 'o-', color='green', label="Cross-validation Score")
        axes[1].legend()

        # Build ROC Curve
        best_VC_param = param_range[np.argmax(valid_scores_mean_VC)]
        print('Best DT alpha:')
        textOut.write(f'Best DT alpha:\n{best_VC_param}\n')
        print(best_VC_param)
        print('--- --- --- --- --- --- ---')
        bestEst.set_params(**{param_name: best_VC_param})
        y_score = bestEst.predict_proba(ss.transform(X_test))[:,1]
        fpr, tpr , t = roc_curve(y_test, y_score)
        auc_scr = roc_auc_score(y_test, y_score)
        axes[2].set_title("Receiver Operating Characteristic")
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].plot(fpr, tpr, color='orange', label=f'ROC Curve (area = {auc_scr})')
        axes[2].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axes[2].legend(loc="lower right")

        print('DT confusion matrix')
        conMat = confusion_matrix(y_test, (y_score>0.5)*1)
        print(conMat)
        textOut.write(f'DT confusion matrix:\n{conMat}\n')
        print('--- --- --- --- --- --- ---')
        print('DT Classification Report')
        claRep = classification_report(y_test, (y_score>0.5)*1)
        print(claRep)
        textOut.write(f'DT Classification Report:\n{claRep}')
        fig.set_size_inches((20, 5))
        plt.savefig(os.path.join(OUTFILE, DSNAMES[ds_i], 'DTgraphs.svg'), dpi = 400, format='svg')
        textOut.close()
        # plt.show()

    if 'SVM' in algos:
        print()
        print('Running Support Vector Machine...')
        print('--- --- --- --- --- --- ---')
        textOut = open(os.path.join(OUTFILE, DSNAMES[ds_i], 'SVMresults.txt'), 'w')
        param_grid = {'C': [100,200,500,1000,2000,5000,7000,9000,11000],
                      'kernel': ['poly','rbf']}
        mlp = make_pipeline(StandardScaler(), GridSearchCV(SVC(random_state=SEED, probability=True), scoring='f1', param_grid=param_grid, n_jobs=THREADS, refit=True, verbose=VERBOSE))
        mlp.fit(X_train, y_train)
        ss = mlp['standardscaler']
        grid = mlp['gridsearchcv']
        bestEst = grid.best_estimator_
        bestParams = grid.best_params_
        print('Best SVM params:')
        textOut.write(f'Best SVM params:\n{grid.best_params_}\n')
        print(grid.best_params_)
        print('--- --- --- --- --- --- ---')
        fig, axes = plt.subplots(1, 3)
        fig.suptitle('Support Vector Machine Graphs')
        # Build Learning Curve
        train_sizes_LC, train_scores_LC, valid_scores_LC = learning_curve(
            bestEst, X_train, y_train, n_jobs=THREADS, train_sizes=np.linspace(0.1, 1.0, 10), verbose=VERBOSE)
        axes[0].set_title('Learning Curve')
        axes[0].set_xlabel("Training Samples")
        axes[0].set_ylabel("Score")
        train_scores_mean_LC, valid_scores_mean_LC = train_scores_LC.mean(axis=1), valid_scores_LC.mean(axis=1)
        axes[0].plot(train_sizes_LC, train_scores_mean_LC, 'o-', color='red', label="Training Score")
        axes[0].plot(train_sizes_LC, valid_scores_mean_LC, 'o-', color='green', label="Cross-validation Score")
        axes[0].legend()

        # Build Validation Curve
        param_range = np.logspace(-10, -3, 25)
        param_name = "gamma"
        train_scores_VC, valid_scores_VC = validation_curve(
            bestEst, X_train, y_train, param_name=param_name, param_range=param_range, scoring="f1", n_jobs=THREADS, verbose=VERBOSE)
        axes[1].set_title('Complexity/Validation Curve')
        axes[1].set_xlabel(r"$\gamma$")
        axes[1].set_ylabel("Score")
        train_scores_mean_VC, valid_scores_mean_VC = train_scores_VC.mean(axis=1), valid_scores_VC.mean(axis=1)
        axes[1].semilogx(param_range, train_scores_mean_VC, 'o-', color='red', label="Training Score")
        axes[1].semilogx(param_range, valid_scores_mean_VC, 'o-', color='green', label="Cross-validation Score")
        axes[1].legend()

        # Build ROC Curve
        best_VC_param = param_range[np.argmax(valid_scores_mean_VC)]
        print('Best SVM gamma:')
        textOut.write(f'Best SVM gamma:\n{best_VC_param}\n')
        print(best_VC_param)
        print('--- --- --- --- --- --- ---')
        bestEst.set_params(**{param_name: best_VC_param})
        y_score = bestEst.predict_proba(ss.transform(X_test))[:,1]

        fpr, tpr , _ = roc_curve(y_test, y_score)
        auc_scr = roc_auc_score(y_test, y_score)
        axes[2].set_title("Receiver Operating Characteristic")
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].plot(fpr, tpr, color='orange', label=f'ROC Curve (area = {auc_scr})')
        axes[2].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axes[2].legend(loc="lower right")

        print('SVM confusion matrix')
        conMat = confusion_matrix(y_test, (y_score>0.5)*1)
        print(conMat)
        textOut.write(f'SVM confusion matrix:\n{conMat}\n')
        print('--- --- --- --- --- --- ---')
        print('SVM Classification Report')
        claRep = classification_report(y_test, (y_score>0.5)*1)
        print(claRep)
        textOut.write(f'SVM Classification Report:\n{claRep}')
        fig.set_size_inches((20, 5))
        plt.savefig(os.path.join(OUTFILE, DSNAMES[ds_i], 'SVMgraphs.svg'), dpi = 400, format='svg')
        textOut.close()
        # plt.show()

    if 'kNN' in algos:
        print()
        print('Running k-Nearest Neighbors...')
        print('--- --- --- --- --- --- ---')
        textOut = open(os.path.join(OUTFILE, DSNAMES[ds_i], 'kNNresults.txt'), 'w')
        param_grid = {'weights': ['uniform', 'distance'],
                      'metric': ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra']}
        mlp = make_pipeline(StandardScaler(), GridSearchCV(KNeighborsClassifier(), scoring='f1', param_grid=param_grid, n_jobs=THREADS, refit=True, verbose=VERBOSE))
        mlp.fit(X_train, y_train)
        ss = mlp['standardscaler']
        grid = mlp['gridsearchcv']
        bestEst = grid.best_estimator_
        bestParams = grid.best_params_
        print('Best kNN params:')
        textOut.write(f'Best kNN params:\n{grid.best_params_}\n')
        print(grid.best_params_)
        print('--- --- --- --- --- --- ---')

        fig, axes = plt.subplots(1, 3)
        fig.suptitle('k-Nearest Neighbor Graphs')
        # Build Learning Curve
        train_sizes_LC, train_scores_LC, valid_scores_LC = learning_curve(
            bestEst, X_train, y_train, n_jobs=THREADS, train_sizes=np.linspace(0.1, 1.0, 10))
        axes[0].set_title('Learning Curve')
        axes[0].set_xlabel("Training Samples")
        axes[0].set_ylabel("Score")
        train_scores_mean_LC, valid_scores_mean_LC = train_scores_LC.mean(axis=1), valid_scores_LC.mean(axis=1)
        axes[0].plot(train_sizes_LC, train_scores_mean_LC, 'o-', color='red', label="Training Score")
        axes[0].plot(train_sizes_LC, valid_scores_mean_LC, 'o-', color='green', label="Cross-validation Score")
        axes[0].legend()

        # Build Validation Curve
        param_range = np.arange(2,50)
        param_name = "n_neighbors"
        train_scores_VC, valid_scores_VC = validation_curve(
            bestEst, X_train, y_train, param_name=param_name, param_range=param_range, scoring="f1", n_jobs=THREADS)
        axes[1].set_title('Complexity/Validation Curve')
        axes[1].set_xlabel(r"Number of Neighbors")
        axes[1].set_ylabel("Score")
        train_scores_mean_VC, valid_scores_mean_VC = train_scores_VC.mean(axis=1), valid_scores_VC.mean(axis=1)
        axes[1].plot(param_range, train_scores_mean_VC, 'o-', color='red', label="Training Score")
        axes[1].plot(param_range, valid_scores_mean_VC, 'o-', color='green', label="Cross-validation Score")
        axes[1].legend()

        # Build ROC Curve
        best_VC_param = param_range[np.argmax(valid_scores_mean_VC)]
        print('Best N Neighbors:')
        textOut.write(f'Best N Neighbors:\n{best_VC_param}\n')
        print(best_VC_param)
        print('--- --- --- --- --- --- ---')
        bestEst.set_params(**{param_name: best_VC_param})
        y_score = bestEst.predict_proba(ss.transform(X_test))[:,1]
        fpr, tpr , _ = roc_curve(y_test, y_score)
        auc_scr = roc_auc_score(y_test, y_score)
        axes[2].set_title("Receiver Operating Characteristic")
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].plot(fpr, tpr, color='orange', label=f'ROC Curve (area = {auc_scr})')
        axes[2].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axes[2].legend(loc="lower right")

        print('kNN confusion matrix')
        conMat = confusion_matrix(y_test, (y_score>0.5)*1)
        print(conMat)
        textOut.write(f'kNN confusion matrix:\n{conMat}\n')
        print('--- --- --- --- --- --- ---')
        print('kNN Classification Report')
        claRep = classification_report(y_test, (y_score>0.5)*1)
        print(claRep)
        textOut.write(f'kNN Classification Report:\n{claRep}')
        fig.set_size_inches((20, 5))
        plt.savefig(os.path.join(OUTFILE, DSNAMES[ds_i], 'kNNgraphs.svg'), dpi = 400, format='svg')
        textOut.close()
        # plt.show()

    if 'ANN' in algos:
        print()
        print('Running Neural Network...')
        print('--- --- --- --- --- --- ---')
        textOut = open(os.path.join(OUTFILE, DSNAMES[ds_i], 'ANNresults.txt'), 'w')
        param_grid = {'hidden_layer_sizes': [(5,), (5,5,)],
                      'activation': ['logistic', 'tanh', 'relu'],
                      'alpha': np.logspace(-5, -2, 25)}
        mlp = make_pipeline(StandardScaler(), GridSearchCV(MLPClassifier(random_state=SEED, max_iter=1000), scoring='f1', param_grid=param_grid, n_jobs=THREADS, refit=True, verbose=VERBOSE))
        mlp.fit(X_train, y_train)
        ss = mlp['standardscaler']
        grid = mlp['gridsearchcv']
        bestEst = grid.best_estimator_
        bestParams = grid.best_params_
        print('Best ANN params:')
        textOut.write(f'Best ANN params:\n{grid.best_params_}\n')
        print(grid.best_params_)
        print('--- --- --- --- --- --- ---')

        fig, axes = plt.subplots(1, 3)
        fig.suptitle('Neural Network Graphs')
        # Build Learning Curve
        train_sizes_LC, train_scores_LC, valid_scores_LC = learning_curve(
            bestEst, X_train, y_train, n_jobs=THREADS, train_sizes=np.linspace(0.1, 1.0, 10))
        axes[0].set_title('Learning Curve')
        axes[0].set_xlabel("Training Samples")
        axes[0].set_ylabel("Score")
        train_scores_mean_LC, valid_scores_mean_LC = train_scores_LC.mean(axis=1), valid_scores_LC.mean(axis=1)
        axes[0].plot(train_sizes_LC, train_scores_mean_LC, 'o-', color='red', label="Training Score")
        axes[0].plot(train_sizes_LC, valid_scores_mean_LC, 'o-', color='green', label="Cross-validation Score")
        axes[0].legend()

        # Build Validation Curve
        a = [(i,) for i in range(3,8)]
        param_range = [sum(i,()) for i in product(a,repeat=len(bestParams['hidden_layer_sizes']))]
        totNodes = np.sum(param_range, axis=1)
        sortIdx = np.argsort(totNodes)
        param_name = "hidden_layer_sizes"
        train_scores_VC, valid_scores_VC = validation_curve(
            bestEst, X_train, y_train, param_name=param_name, param_range=param_range, scoring="f1", n_jobs=THREADS)
        axes[1].set_title('Complexity/Validation Curve')
        axes[1].set_xlabel(r"Structure - Number of Nodes")
        axes[1].set_ylabel("Score")
        train_scores_mean_VC, valid_scores_mean_VC = train_scores_VC.mean(axis=1), valid_scores_VC.mean(axis=1)
        tickIDs = np.arange(len(param_range))
        axes[1].plot(tickIDs, train_scores_mean_VC[sortIdx], 'o-', color='red', label="Training Score")
        axes[1].plot(tickIDs, valid_scores_mean_VC[sortIdx], 'o-', color='green', label="Cross-validation Score")
        axes[1].set_xticks(tickIDs)
        axes[1].set_xticklabels([i[0]+ ' - ' + i[1] for i in zip(map(str, map(tuple, np.array(param_range)[sortIdx])), totNodes.astype(str)[sortIdx])], rotation=-85)
        axes[1].legend()

        # Build ROC Curve
        best_VC_param = param_range[np.argmax(valid_scores_mean_VC)]
        print('Best ANN structure:')
        textOut.write(f'Best ANN structure:\n{best_VC_param}\n')
        print(best_VC_param)
        print('--- --- --- --- --- --- ---')
        bestEst.set_params(**{param_name: best_VC_param})
        y_score = bestEst.predict_proba(ss.transform(X_test))[:,1]
        fpr, tpr , _ = roc_curve(y_test, y_score)
        auc_scr = roc_auc_score(y_test, y_score)
        axes[2].set_title("Receiver Operating Characteristic")
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].plot(fpr, tpr, color='orange', label=f'ROC Curve (area = {auc_scr})')
        axes[2].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axes[2].legend(loc="lower right")

        print('ANN confusion matrix')
        conMat = confusion_matrix(y_test, (y_score>0.5)*1)
        print(conMat)
        textOut.write(f'ANN confusion matrix:\n{conMat}\n')
        print('--- --- --- --- --- --- ---')
        print('ANN Classification Report')
        claRep = classification_report(y_test, (y_score>0.5)*1)
        print(claRep)
        textOut.write(f'ANN Classification Report:\n{claRep}')
        fig.set_size_inches((20, 5))
        plt.savefig(os.path.join(OUTFILE, DSNAMES[ds_i], 'ANNgraphs.svg'), dpi = 400, format='svg')
        textOut.close()
        # plt.show()

    if 'BDT' in algos:
        print()
        print('Running Boosting...')
        print('--- --- --- --- --- --- ---')
        textOut = open(os.path.join(OUTFILE, DSNAMES[ds_i], 'BDTresults.txt'), 'w')
        param_grid = {'n_estimators': np.arange(50,300,50),
                      'max_depth': np.arange(1, 21)}
        mlp = make_pipeline(StandardScaler(), GridSearchCV(GradientBoostingClassifier(random_state=SEED), scoring='f1', param_grid=param_grid, n_jobs=THREADS, refit=True, verbose=VERBOSE))
        mlp.fit(X_train, y_train)
        ss = mlp['standardscaler']
        grid = mlp['gridsearchcv']
        bestEst = grid.best_estimator_
        bestParams = grid.best_params_
        print('Best BDT params:')
        textOut.write(f'Best BDT params:\n{grid.best_params_}\n')
        print(grid.best_params_)
        print('--- --- --- --- --- --- ---')

        fig, axes = plt.subplots(1,3)
        fig.suptitle('Boosting (Gradient Tree Boosting) Graphs')
        # Build Learning Curve
        train_sizes_LC, train_scores_LC, valid_scores_LC = learning_curve(
            bestEst, X_train, y_train, n_jobs=THREADS, train_sizes=np.linspace(0.1, 1.0, 10))
        axes[0].set_title('Learning Curve')
        axes[0].set_xlabel("Training Samples")
        axes[0].set_ylabel("Score")
        train_scores_mean_LC, valid_scores_mean_LC = train_scores_LC.mean(axis=1), valid_scores_LC.mean(axis=1)
        axes[0].plot(train_sizes_LC, train_scores_mean_LC, 'o-', color='red', label="Training Score")
        axes[0].plot(train_sizes_LC, valid_scores_mean_LC, 'o-', color='green', label="Cross-validation Score")
        axes[0].legend()

        # Build Validation Curve
        param_range = np.logspace(-4, 2, 30)
        param_name = "ccp_alpha"
        train_scores_VC, valid_scores_VC = validation_curve(
            bestEst, X_train, y_train, param_name=param_name, param_range=param_range, scoring="f1", n_jobs=THREADS)
        axes[1].set_title('Complexity/Validation Curve')
        axes[1].set_xlabel(r"$\alpha$")
        axes[1].set_ylabel("Score")
        train_scores_mean_VC, valid_scores_mean_VC = train_scores_VC.mean(axis=1), valid_scores_VC.mean(axis=1)
        axes[1].semilogx(param_range, train_scores_mean_VC, 'o-', color='red', label="Training Score")
        axes[1].semilogx(param_range, valid_scores_mean_VC, 'o-', color='green', label="Cross-validation Score")
        axes[1].legend()

        # Build ROC Curve
        best_VC_param = param_range[np.argmax(valid_scores_mean_VC)]
        print('Best BDT alpha:')
        textOut.write(f'Best BDT alpha:\n{best_VC_param}\n')
        print(best_VC_param)
        print('--- --- --- --- --- --- ---')
        bestEst.set_params(**{param_name: best_VC_param})
        y_score = bestEst.predict_proba(ss.transform(X_test))[:,1]
        fpr, tpr , _ = roc_curve(y_test, y_score)
        auc_scr = roc_auc_score(y_test, y_score)
        axes[2].set_title("Receiver Operating Characteristic")
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].plot(fpr, tpr, color='orange', label=f'ROC Curve (area = {auc_scr})')
        axes[2].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axes[2].legend(loc="lower right")

        print('BDT confusion matrix')
        conMat = confusion_matrix(y_test, (y_score>0.5)*1)
        print(conMat)
        textOut.write(f'BDT confusion matrix:\n{conMat}\n')
        print('--- --- --- --- --- --- ---')
        print('BDT Classification Report')
        claRep = classification_report(y_test, (y_score>0.5)*1)
        print(claRep)
        textOut.write(f'BDT Classification Report:\n{claRep}')
        fig.set_size_inches((20, 5))
        plt.savefig(os.path.join(OUTFILE, DSNAMES[ds_i], 'BDTgraphs.svg'), dpi = 400, format='svg')
        textOut.close()
        # plt.show()
