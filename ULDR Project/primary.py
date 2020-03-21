# Imports
import numpy as np
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.random_projection import SparseRandomProjection as SRP
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, homogeneity_completeness_v_measure
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import pandas as pd
import os
import sys
import pickle
import shutil
from pathlib import Path
SEED = 10
THREADS = 8
OUTFILE = r"Results"
VERBOSE = 0
np.random.seed(SEED)

# ---------------------------------------------------------------
def kMeans(k, X):
    est = KMeans(n_clusters=k,
                 random_state=SEED, n_jobs=THREADS)
    labels = est.fit_predict(X)
    silavgint = silhouette_score(X, labels)
    return (silavgint, est)

def EM(k, X, cv_type):
    est = GaussianMixture(n_components=k,
                          covariance_type=cv_type,
                          max_iter=300,
                          n_init=10,
                          init_params='random',
                          random_state=SEED)
    est.fit(X)
    return (-est.bic(X), est)

def handle_clusters(X, outpath, name=''):
    CovTypes = ['spherical', 'tied', 'diag', 'full']
    k_vals = np.arange(2, 11)
    silavg = np.zeros_like(k_vals).astype(float)
    BICs = np.repeat(np.zeros_like(k_vals)[None, :], 4, axis=0).astype(float)
    best_sa = -np.inf
    best_bic = -np.inf
    for i, k in enumerate(k_vals):
        print(f'Number of Clusters: {k}')
        s, kme = kMeans(k, X)
        silavg[i] = s
        if s > best_sa:
            best_sa = s
            best_KM = kme
        for j, cv_type in enumerate(CovTypes):
            print(f'Covariance Type: {cv_type}')
            b, gmm = EM(k, X, cv_type)
            BICs[j, i] = b
            if b > best_bic:
                best_bic = b
                best_EM = gmm
    fig, ax = plt.subplots(1, 2)
    plt.title('Cluster Selection')
    ax[0].set_title('Silhouette Scores for \n Cluster Selection in K-Means')
    ax[0].set_xlabel('# Clusters')
    ax[0].set_ylabel('Silhouette Score')
    ax[0].plot(k_vals, silavg)
    ax[0].scatter(k_vals[np.argmax(silavg)], np.max(silavg))
    ax[1].set_title('BIC Scores for Cluster Selection \n in Gaussian Mixture Models')
    ax[1].set_xlabel('# Clusters')
    ax[1].set_ylabel('BIC Score')
    if not np.any(BICs < 0):
        ax[1].set_yscale('log')
        ax[1].set_ylabel('Log of BIC Score')
    for l, bic in enumerate(BICs):
        ax[1].plot(k_vals, bic, label=CovTypes[l])
        ax[1].scatter(k_vals[np.argmax(bic)], np.max(bic))
    ax[1].legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(outpath, name + 'Cluster Selection.png'), dpi=400, format='png')
    plt.close(fig)
    return (best_KM, best_EM)

def handle_cluster_visualization(est, X, y, outpath, gName=''):
    try:
        n_c = np.min([est.n_clusters, min(*X.shape)])
    except:
        n_c = np.min([est.n_components, min(*X.shape)])
    try:
        labels = est.labels_
        flag = False
        name = 'K-Means'
    except:
        labels = est.predict(X)
        flag = True
        name = 'GMM'

    textOut = open(os.path.join(outpath, gName + f'{name}_clusterAnalysis.txt'), 'w')
    H, C, V = homogeneity_completeness_v_measure(y, labels)
    textOut.write(f'Homogeneity: {H}\n')
    textOut.write(f'Completeness: {C}\n')
    textOut.write(f'V-Measure: {V}\n')
    textOut.close()
    pca = PCA(n_components=n_c)
    pca_result = pca.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(pca_result)

    if n_c > 2:
        fig = plt.figure()
        ax_pca = Axes3D(fig)
        def init():
            ax_pca.set_title(f'PCA Component Projections for {name}')
            for cl in np.unique(labels):
                mask = labels == cl
                ax_pca.scatter(xs=pca_result[mask][:, 0],
                           ys=pca_result[mask][:, 1],
                           zs=pca_result[mask][:, 2],
                           alpha=0.5, label=cl)
            ax_pca.set_xlabel('pca_one')
            ax_pca.set_ylabel('pca_two')
            ax_pca.set_zlabel('pca_three')
            if flag:
                plt.legend(title='Labels')
            else:
                plt.legend(title='Clusters')
            return fig,

        def animate(i):
            ax_pca.view_init(30, i)
            return fig,

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20)
        anim.save(os.path.join(outpath, gName + f'{name} PCA Projection (Animated).gif'), writer='imagemagick', fps=20, dpi=100)
        plt.close()

    else:
        ax_pca = plt.figure().gca()
        ax_pca.set_title(f'PCA Component Projections for {name}')
        for cl in np.unique(labels):
            mask = labels == cl
            ax_pca.scatter(x=pca_result[mask][:, 0], y=pca_result[mask][:, 1], alpha=0.5, label=cl)
        ax_pca.set_xlabel('pca_one')
        ax_pca.set_ylabel('pca_two')
        if flag:
            plt.legend(title='Labels')
        else:
            plt.legend(title='Clusters')
        # plt.show()
        plt.savefig(os.path.join(outpath, gName + f'{name} PCA Projection.png'), dpi=400, format='png')
        plt.close()

    ax_tsne = plt.figure().gca()
    ax_tsne.set_title(f't-SNE Component Projections for {name}')
    for cl in np.unique(labels):
        mask = labels == cl
        ax_tsne.scatter(x=tsne_result[mask][:, 0], y=tsne_result[mask][:, 1], alpha=0.5, label=cl)

    ax_tsne.set_xlabel('tsne_one')
    ax_tsne.set_ylabel('tsne_two')
    if flag:
        plt.legend(title='Labels')
    else:
        plt.legend(title='Clusters')
    # plt.show()
    plt.savefig(os.path.join(outpath, gName + f'{name} t-SNE Projection.png'), dpi=400, format='png')
    plt.close()

def handle_dimredux(X, outpath, PCA_cut=0.95, SVD_cut=0.95):
    # PCA
    pca = PCA(PCA_cut, whiten=True, svd_solver='auto', random_state=SEED)
    pcaRes = pca.fit_transform(X)
    # pcaRes = (pca.fit_transform(X), pca)
    plt.plot(pca.explained_variance_)
    plt.xlabel('Component')
    plt.ylabel('Eigenvalues')
    plt.title(f'Distribution of Eigenvalues over PCA components \n Explains {PCA_cut * 100}% of Variance, k={pca.n_components}')
    plt.savefig(os.path.join(outpath, 'PCAEigenDist.png'), dpi=400, format='png')
    plt.close()

    # ICA
    ica = FastICA(whiten=True, random_state=SEED, max_iter=10000, tol=0.001)
    ica.fit(X)
    kvals = []
    xv = np.arange(2, ica.components_.shape[0])
    for i in xv:
        kvals.append(np.mean(kurtosis(np.dot(X, ica.components_[:i].T))**2))  # transform X with increasingly more ICA components and calculate the kurtosis of the transformation
    ica_k = xv[np.argmax(kvals)]
    icaRes = np.dot(X, ica.components_[:ica_k].T)  # Take the X transform with greatest kurtosis
    # icaRes = (np.dot(X, ica.components_[:ica_k].T), ica)  # Take the X transform with greatest kurtosis
    plt.plot(xv, kvals, label='Kurtosis')
    plt.vlines(ica_k, 0, np.max(kvals), label=f'Best K: {ica_k}')
    plt.yscale('log')
    plt.xlabel('ICA Components')
    plt.ylabel('Mean Squared Kurtosis')
    plt.title('Kurtosis of ICA Components')
    plt.legend()
    plt.savefig(os.path.join(outpath, 'ICAKurtosis.png'), dpi=400, format='png')
    plt.close()

    # RCA
    reconScore = []
    X_ts = []
    Xvals = np.arange(2, X.shape[1])
    for i in Xvals:
        rca = SRP(i, dense_output=True)
        X_t = rca.fit_transform(X)
        reverse = np.linalg.pinv(rca.components_.toarray())
        l = 0
        for j in range(9):
            rca = SRP(i, dense_output=True)
            X_t += rca.fit_transform(X)
            reverse += np.linalg.pinv(rca.components_.toarray())
            l += 1
        reconScore.append(((X - np.dot(X_t / (1 + l), reverse.T / (1 + l)))**2).mean())
        X_ts.append(X_t / (1 + l))
    rca_k = Xvals[np.argmin(reconScore)]
    if rca_k > len(reconScore):
        rca_k = len(reconScore) -1
    minError = reconScore[rca_k]
    rcaRes = X_ts[rca_k]
    # rcaRes = (X_ts[rca_k], None)
    plt.plot(Xvals, reconScore, label='Recon. Score')
    plt.vlines(rca_k, 0, max(reconScore), label=f'Best K: {rca_k}')
    plt.title('Reconstruction Scores (MSE) for Randomized Projections')
    plt.xlabel('Components')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(os.path.join(outpath, 'RCARecon.png'), dpi=400, format='png')
    plt.close()

    # SVD
    svd = TruncatedSVD(X.shape[1] - 1, random_state=SEED)
    svd.fit(X)
    evr_Cumm = np.cumsum(svd.explained_variance_ratio_)
    svd_k = (evr_Cumm <= SVD_cut).sum()
    svdRes = svd.transform(X)
    # svdRes = (svd.transform(X), svd)
    plt.plot(evr_Cumm, label='Cumm. Ratio')
    plt.plot(svd.explained_variance_ratio_, label='Ratio of exp. var.')
    plt.vlines(svd_k, 0, 1, label=f'Best K: {svd_k}')
    plt.title(f'Choosing best k components for Truncated SVD \n Explains {SVD_cut * 100}% of variance')
    plt.xlabel('Components')
    plt.ylabel('Ratio/Percentage of Explained Variance')
    plt.legend()
    plt.savefig(os.path.join(outpath, 'SVDChooseK.png'), dpi=400, format='png')
    plt.close()

    return pcaRes, icaRes, rcaRes, svdRes

def handle_NN(X, y, outpath, name):
    textOut = open(os.path.join(outpath, f'{name}_ANNresults.txt'), 'w')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=SEED)
    clf = MLPClassifier(hidden_layer_sizes=(7,5),
                        activation='relu',
                        alpha=0.00237,
                        max_iter=5000,
                        random_state=SEED)

    clf.fit(X_train, y_train)

    figLC, axLC = plt.subplots()
    # Build Learning Curve
    train_sizes_LC, train_scores_LC, valid_scores_LC = learning_curve(
        clf, X_train, y_train, n_jobs=THREADS, train_sizes=np.linspace(0.1, 1.0, 10))
    axLC.set_title('Learning Curve')
    axLC.set_xlabel("Training Samples")
    axLC.set_ylabel("Score")
    train_scores_mean_LC, valid_scores_mean_LC = train_scores_LC.mean(axis=1), valid_scores_LC.mean(axis=1)
    axLC.plot(train_sizes_LC, train_scores_mean_LC, 'o-', color='red', label="Training Score")
    axLC.plot(train_sizes_LC, valid_scores_mean_LC, 'o-', color='green', label="Cross-validation Score")
    axLC.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, f'{name}_LCGraph.png'), dpi = 400, format='png')
    plt.close(figLC)

    # Build ROC Curve
    figRoc, axRoc = plt.subplots()
    y_score = clf.predict_proba(X_test)[:,1]
    fpr, tpr , _ = roc_curve(y_test, y_score)
    auc_scr = roc_auc_score(y_test, y_score)
    axRoc.set_title("Receiver Operating Characteristic")
    axRoc.set_xlabel('False Positive Rate')
    axRoc.set_ylabel('True Positive Rate')
    axRoc.plot(fpr, tpr, color='orange', label=f'ROC Curve (area = {auc_scr})')
    axRoc.plot([0, 1], [0, 1], color='navy', linestyle='--')
    axRoc.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, f'{name}_ROCGraph.png'), dpi = 400, format='png')
    plt.close(figRoc)

    y_predict = clf.predict(X_test)
    conMat = confusion_matrix(y_test, y_predict)
    textOut.write(f'ANN confusion matrix:\n{conMat}\n')
    claRep = classification_report(y_test, y_predict)
    textOut.write(f'ANN Classification Report:\n{claRep}')
    textOut.close()

# ---------------------------------------------------------------

# Data import and preprocessing
DS_1_fName = r"Data\Epileptic Seizure Recognition\PPData.csv"
DS_1 = pd.read_csv(os.path.join(sys.path[0], DS_1_fName)).to_numpy()[:, 1:].astype(int)
np.random.shuffle(DS_1)
DS_1_X = DS_1[:, :-1]
DS_1_Y = DS_1[:, -1]

DS_2_fName = r"Data\MAGIC Gamma Telescope\PPData.csv"
DS_2 = pd.read_csv(os.path.join(sys.path[0], DS_2_fName)).to_numpy()[:, 1:].astype(int)
np.random.shuffle(DS_2)
DS_2_X = DS_2[:, :-1]
DS_2_Y = DS_2[:, -1]

DSNAMES = ['Epileptic Seizure Recognition', 'MAGIC Gamma Telescope']
# DS = [(DS_1_X[:int(len(DS_1)/20)],DS_1_Y[:int(len(DS_1)/20)]), (DS_2_X[:int(len(DS_1)/20)],DS_2_Y[:int(len(DS_1)/20)])]
DS = [(DS_1_X,DS_1_Y), (DS_2_X,DS_2_Y)]

if os.path.exists(OUTFILE):
    if input('Fresh Run? >') == 'y':
        shutil.rmtree(OUTFILE, True)

print('Choose Scaling Method:'
print('{RobustScaler: "rs", StandardScaler: "ss", QuantileTransformer: "qt"}')
while True:
    scaler = input('>> ')
    if scaler not in ['rs','ss','qt']:
        print('Please choose a scaling method as "rs", "ss", or "qt"')
    else:
        break

for ds_i, ds in enumerate(DS):
    if input(f'Run {DSNAMES[ds_i]}? >') == 'y':
        X, y = ds
        if scaler == 'rs':
            scale = RobustScaler().fit(X)
        elif scaler == 'ss':
            scale = StandardScaler().fit(X)
        elif scaler == 'qt':
            scale = QuantileTransformer(n_quantiles=np.min([1000, X.shape[0]]), output_distribution='uniform').fit(X)
        else:
            raise ValueError, 'Improper scaling method chosen'
        X = scale.transform(X)
        Path(os.path.join(OUTFILE, DSNAMES[ds_i])).mkdir(parents=True, exist_ok=True)
        outpath = os.path.join(OUTFILE, DSNAMES[ds_i])
        print(f'Running {DSNAMES[ds_i]}\n')

        # Part 1 - Cluster data
        print('Running Clustering')
        if not (os.path.isfile(os.path.join(outpath, 'KM_est.pkl')) and os.path.isfile(os.path.join(outpath, 'EM_est.pkl'))):
            KM_est, EM_est = handle_clusters(X, outpath)
            handle_cluster_visualization(KM_est, X, y, outpath)
            handle_cluster_visualization(EM_est, X, y, outpath)
            with open(os.path.join(outpath, 'KM_est.pkl'), 'wb') as kmpk:
                pickle.dump(KM_est, kmpk, -1)
            with open(os.path.join(outpath, 'EM_est.pkl'), 'wb') as empk:
                pickle.dump(EM_est, empk, -1)
        else:
            with open(os.path.join(outpath, 'KM_est.pkl'), 'rb') as kmpk:
                KM_est = pickle.load(kmpk)
            with open(os.path.join(outpath, 'EM_est.pkl'), 'rb') as empk:
                EM_est = pickle.load(empk)
        print('Done.\n\n')

        # Part 2 - DimRedux data
        print('Running Dimensional Reduction')
        a = os.path.isfile(os.path.join(outpath, 'DR_PCA.npy'))
        b = os.path.isfile(os.path.join(outpath, 'DR_ICA.npy'))
        c = os.path.isfile(os.path.join(outpath, 'DR_RCA.npy'))
        d = os.path.isfile(os.path.join(outpath, 'DR_SVD.npy'))
        if not (a and b and c and d):
            PCARes, ICARes, RCARes, SVDRes = handle_dimredux(X, outpath)  #Outputs numpy arrays
            np.save(os.path.join(outpath, 'DR_PCA.npy'), PCARes)
            np.save(os.path.join(outpath, 'DR_ICA.npy'), ICARes)
            np.save(os.path.join(outpath, 'DR_RCA.npy'), RCARes)
            np.save(os.path.join(outpath, 'DR_SVD.npy'), SVDRes)
        else:
            PCARes = np.load(os.path.join(outpath, 'DR_PCA.npy'))
            ICARes = np.load(os.path.join(outpath, 'DR_ICA.npy'))
            RCARes = np.load(os.path.join(outpath, 'DR_RCA.npy'))
            SVDRes = np.load(os.path.join(outpath, 'DR_SVD.npy'))
        print('Done.\n\n')

        # Part 3 - Cluster on DimRedux'd data
        if input('Run DimRedux Clustering? >') == 'y':
            print('Clustering on Dim. Reduced Data')
            print('\tPCA....')
            PCA_KM_est, PCA_EM_est = handle_clusters(PCARes, outpath, 'PCA ')
            handle_cluster_visualization(PCA_KM_est, PCARes, y, outpath, 'PCA ')
            handle_cluster_visualization(PCA_EM_est, PCARes, y, outpath, 'PCA ')
            print('\tPCA Done.\n')

            print('\tICA....')
            ICA_KM_est, ICA_EM_est = handle_clusters(ICARes, outpath, 'ICA ')
            handle_cluster_visualization(ICA_KM_est, ICARes, y, outpath, 'ICA ')
            handle_cluster_visualization(ICA_EM_est, ICARes, y, outpath, 'ICA ')
            print('\tICA Done.\n')

            print('\tRCA....')
            RCA_KM_est, RCA_EM_est = handle_clusters(RCARes, outpath, 'RCA ')
            handle_cluster_visualization(RCA_KM_est, RCARes, y, outpath, 'RCA ')
            handle_cluster_visualization(RCA_EM_est, RCARes, y, outpath, 'RCA ')
            print('\tRCA Done.\n')

            print('\tSVD....')
            SVD_KM_est, SVD_EM_est = handle_clusters(SVDRes, outpath, 'SVD ')
            handle_cluster_visualization(SVD_KM_est, SVDRes, y, outpath, 'SVD ')
            handle_cluster_visualization(SVD_EM_est, SVDRes, y, outpath, 'SVD ')
            print('\tSVD Done.\n\n')

        ## NN Parts
        if ds_i == 0:
            if input('Run NN Experiments? >') == 'y':
                print('Running NN components...\n')
                # Part 4 - NN on DimRedux'd data
                print('Running NN on Dim. Reduced Data...')
                handle_NN(PCARes, y, outpath, 'PCA')
                handle_NN(ICARes, y, outpath, 'ICA')
                handle_NN(RCARes, y, outpath, 'RCA')
                handle_NN(SVDRes, y, outpath, 'SVD')
                print('Done.\n')
                # Part 5 - NN with Clusters added
                print('Running NN on base data with added Clusters...\n')
                KM_X = np.hstack([X, np.atleast_2d(KM_est.predict(X)).T])
                handle_NN(KM_X, y, outpath, 'KM')
                EM_X = np.hstack([X, EM_est.predict_proba(X)])
                handle_NN(EM_X, y, outpath, 'GMM')
                print('Done.\n\n')

print('End of Experiments.')
