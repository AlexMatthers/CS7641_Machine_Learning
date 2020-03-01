import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
RANDOM_STATE = 10
THREADS = 12
INPUT_DIRECTORY = os.path.join(sys.path[0],'Output')
OUTPUT_DIRECTORY = os.path.join(sys.path[0],'Output', 'Graphs')
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

SMALL_SIZE = 14
MEDIUM_SIZE = 24
BIGGER_SIZE = 36

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
np.random.seed(RANDOM_STATE)

# Average fitness per iterations (average across paramcombos)
# Average time per iteration per complexity (extend fill stats_df and average across paramcombos)
# Plenty of graphs (36!)
lines = [(0,(1,1)), (0,(5,5)), (0,(3,1,1,1)), (0,(3,1,1,1,1,1)), (0, (3,1,3,1,1,1)), 'solid']
cN = [5, 10, 15, 30, 55, 100]
qN = [20, 35, 50, 65, 80, 95]
pN = [5, 10, 15, 30, 55, 100]
X = np.insert(2**np.arange(10), 0, 0)
dX = np.diff(X)
# MaxK-Color

color_ATI_rhc = []
color_ATI_sa = []
color_ATI_ga = []
color_ATI_mimic = []
for i, c in enumerate(cN):
    color_FPI_Fig, color_FPI_ax = plt.subplots()
    rhc_df = pd.read_csv(os.path.join(INPUT_DIRECTORY, f'color_{c}', f'rhc__color_{c}__run_stats_df.csv'))
    rhc_df = rhc_df[~rhc_df.Iteration.isin([1024,2048])]
    SA_df = pd.read_csv(os.path.join(INPUT_DIRECTORY, f'color_{c}', f'sa__color_{c}__run_stats_df.csv'))
    SA_df = SA_df[~SA_df.Iteration.isin([1024,2048])]
    GA_df = pd.read_csv(os.path.join(INPUT_DIRECTORY, f'color_{c}', f'ga__color_{c}__run_stats_df.csv'))
    GA_df = GA_df[~GA_df.Iteration.isin([1024,2048])]
    mimic_df = pd.read_csv(os.path.join(INPUT_DIRECTORY, f'color_{c}', f'mimic__color_{c}__run_stats_df.csv'))
    mimic_df = mimic_df[~mimic_df.Iteration.isin([1024,2048])]

    color_ATI_rhc.append(np.concatenate([np.repeat(A,B) \
                    for A,B in zip(np.array([ds['Time'] \
                      for ds in np.split(rhc_df, len(rhc_df) / len(X))]).mean(0), dX)]).mean())
    color_ATI_sa.append(np.concatenate([np.repeat(A,B) \
                    for A,B in zip(np.array([ds['Time'] \
                      for ds in np.split(SA_df, len(SA_df) / len(X))]).mean(0), dX)]).mean())
    color_ATI_ga.append(np.concatenate([np.repeat(A,B) \
                    for A,B in zip(np.array([ds['Time'] \
                      for ds in np.split(GA_df, len(GA_df) / len(X))]).mean(0), dX)]).mean())
    color_ATI_mimic.append(np.concatenate([np.repeat(A,B) \
                    for A,B in zip(np.array([ds['Time'] \
                      for ds in np.split(mimic_df, len(mimic_df) / len(X))]).mean(0), dX)]).mean())

    avg_fit_rhc = np.array([ds['Fitness'] for ds in np.split(rhc_df, len(rhc_df) / len(X))]).mean(0)
    avg_fit_rhc = avg_fit_rhc.max() - avg_fit_rhc  # Invert
    color_FPI_ax.plot(X, avg_fit_rhc, linestyle=lines[i], color='red', label=f'RHC Color_N={c}')
    avg_fit_SA = np.array([ds['Fitness'] for ds in np.split(SA_df, len(SA_df) / len(X))]).mean(0)
    avg_fit_SA = avg_fit_SA.max() - avg_fit_SA  # Invert
    color_FPI_ax.plot(X, avg_fit_SA, linestyle=lines[i], color='blue', label=f'SA Color_N={c}')
    avg_fit_GA = np.array([ds['Fitness'] for ds in np.split(GA_df, len(GA_df) / len(X))]).mean(0)
    avg_fit_GA = avg_fit_GA.max() - avg_fit_GA  # Invert
    color_FPI_ax.plot(X, avg_fit_GA, linestyle=lines[i], color='green', label=f'GA Color_N={c}')
    avg_fit_mimic = np.array([ds['Fitness'] for ds in np.split(mimic_df, len(mimic_df) / len(X))]).mean(0)
    avg_fit_mimic = avg_fit_mimic.max() - avg_fit_mimic  # Invert
    color_FPI_ax.plot(X, avg_fit_mimic, linestyle=lines[i], color='orange', label=f'MIMIC Color_N={c}')

    color_FPI_ax.set_title(f'Average Fitness for MaxK-Color, N = {c}')
    color_FPI_ax.set_xlabel('Iterations')
    color_FPI_ax.set_ylabel('Fitness')
    # color_FPI_ax.set_yscale('log')
    color_FPI_ax.set_xscale('log')
    color_FPI_ax.legend()
    color_FPI_Fig.set_size_inches((15, 10))
    plt.savefig(os.path.join(OUTPUT_DIRECTORY,f'color_{c}_FPI.png'), dpi=400, format='png')
    plt.close(color_FPI_Fig)

color_ATI_Fig, color_ATI_ax = plt.subplots()
color_ATI_ax.set_title('Analog of Computational Load for MaxK-Color')
color_ATI_ax.set_xlabel('Number of Graph Nodes')
color_ATI_ax.set_ylabel('Average Time / Iteration')
color_ATI_ax.plot(cN, color_ATI_rhc, 'o-', color='red', label='RHC')
color_ATI_ax.plot(cN, color_ATI_sa, 'o-', color='blue', label='SA')
color_ATI_ax.plot(cN, color_ATI_ga, 'o-', color='green', label='GA')
color_ATI_ax.plot(cN, color_ATI_mimic, 'o-', color='orange', label='MIMIC')
color_ATI_ax.set_yscale('log')
color_ATI_ax.legend()
color_ATI_Fig.set_size_inches((10, 10))
color_ATI_Fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIRECTORY,'color_ATI.png'), dpi=400, format='png')
plt.close(color_ATI_Fig)

# Continuous Peaks

peaks_ATI_rhc = []
peaks_ATI_sa = []
peaks_ATI_ga = []
peaks_ATI_mimic = []
for i, c in enumerate(pN):
    peaks_FPI_Fig, peaks_FPI_ax = plt.subplots()
    rhc_df = pd.read_csv(os.path.join(INPUT_DIRECTORY, f'peaks_{c}', f'rhc__peaks_{c}__run_stats_df.csv'))
    rhc_df = rhc_df[~rhc_df.Iteration.isin([1024,2048])]
    SA_df = pd.read_csv(os.path.join(INPUT_DIRECTORY, f'peaks_{c}', f'sa__peaks_{c}__run_stats_df.csv'))
    SA_df = SA_df[~SA_df.Iteration.isin([1024,2048])]
    GA_df = pd.read_csv(os.path.join(INPUT_DIRECTORY, f'peaks_{c}', f'ga__peaks_{c}__run_stats_df.csv'))
    GA_df = GA_df[~GA_df.Iteration.isin([1024,2048])]
    mimic_df = pd.read_csv(os.path.join(INPUT_DIRECTORY, f'peaks_{c}', f'mimic__peaks_{c}__run_stats_df.csv'))
    mimic_df = mimic_df[~mimic_df.Iteration.isin([1024,2048])]

    peaks_ATI_rhc.append(np.concatenate([np.repeat(A,B) \
                    for A,B in zip(np.array([ds['Time'] \
                      for ds in np.split(rhc_df, len(rhc_df) / len(X))]).mean(0), dX)]).mean())
    peaks_ATI_sa.append(np.concatenate([np.repeat(A,B) \
                    for A,B in zip(np.array([ds['Time'] \
                      for ds in np.split(SA_df, len(SA_df) / len(X))]).mean(0), dX)]).mean())
    peaks_ATI_ga.append(np.concatenate([np.repeat(A,B) \
                    for A,B in zip(np.array([ds['Time'] \
                      for ds in np.split(GA_df, len(GA_df) / len(X))]).mean(0), dX)]).mean())
    peaks_ATI_mimic.append(np.concatenate([np.repeat(A,B) \
                    for A,B in zip(np.array([ds['Time'] \
                      for ds in np.split(mimic_df, len(mimic_df) / len(X))]).mean(0), dX)]).mean())

    avg_fit_rhc = np.array([ds['Fitness'] for ds in np.split(rhc_df, len(rhc_df) / len(X))]).mean(0)
    # avg_fit_rhc = avg_fit_rhc.max() - avg_fit_rhc  # Invert
    peaks_FPI_ax.plot(X, avg_fit_rhc, linestyle=lines[i], color='red', label=f'RHC Peaks_N={c}')
    avg_fit_SA = np.array([ds['Fitness'] for ds in np.split(SA_df, len(SA_df) / len(X))]).mean(0)
    # avg_fit_SA = avg_fit_SA.max() - avg_fit_SA  # Invert
    peaks_FPI_ax.plot(X, avg_fit_SA, linestyle=lines[i], color='blue', label=f'SA Peaks_N={c}')
    avg_fit_GA = np.array([ds['Fitness'] for ds in np.split(GA_df, len(GA_df) / len(X))]).mean(0)
    # avg_fit_GA = avg_fit_GA.max() - avg_fit_GA  # Invert
    peaks_FPI_ax.plot(X, avg_fit_GA, linestyle=lines[i], color='green', label=f'GA Peaks_N={c}')
    avg_fit_mimic = np.array([ds['Fitness'] for ds in np.split(mimic_df, len(mimic_df) / len(X))]).mean(0)
    # avg_fit_mimic = avg_fit_mimic.max() - avg_fit_mimic  # Invert
    peaks_FPI_ax.plot(X, avg_fit_mimic, linestyle=lines[i], color='orange', label=f'MIMIC Peaks_N={c}')

    peaks_FPI_ax.set_title(f'Average Fitness for Continuous Peaks, N = {c}')
    peaks_FPI_ax.set_xlabel('Iterations')
    peaks_FPI_ax.set_ylabel('Fitness')
    # peaks_FPI_ax.set_yscale('log')
    peaks_FPI_ax.set_xscale('log')
    peaks_FPI_ax.legend()
    peaks_FPI_Fig.set_size_inches((15, 10))
    plt.savefig(os.path.join(OUTPUT_DIRECTORY,f'peaks_{c}_FPI.png'), dpi=400, format='png')
    plt.close(peaks_FPI_Fig)

peaks_ATI_Fig, peaks_ATI_ax = plt.subplots()
peaks_ATI_ax.set_title('Analog of Computational Load for Continuous Peaks')
peaks_ATI_ax.set_xlabel('Number of Graph Nodes')
peaks_ATI_ax.set_ylabel('Average Time / Iteration')
peaks_ATI_ax.plot(pN, peaks_ATI_rhc, 'o-', color='red', label='RHC')
peaks_ATI_ax.plot(pN, peaks_ATI_sa, 'o-', color='blue', label='SA')
peaks_ATI_ax.plot(pN, peaks_ATI_ga, 'o-', color='green', label='GA')
peaks_ATI_ax.plot(pN, peaks_ATI_mimic, 'o-', color='orange', label='MIMIC')
peaks_ATI_ax.set_yscale('log')
peaks_ATI_ax.legend()
peaks_ATI_Fig.set_size_inches((10, 10))
peaks_ATI_Fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIRECTORY,'peaks_ATI.png'), dpi=400, format='png')
plt.close(peaks_ATI_Fig)

# N-Queens

queens_ATI_rhc = []
queens_ATI_sa = []
queens_ATI_ga = []
queens_ATI_mimic = []
for i, c in enumerate(qN):
    queens_FPI_Fig, queens_FPI_ax = plt.subplots()
    rhc_df = pd.read_csv(os.path.join(INPUT_DIRECTORY, f'queens_{c}', f'rhc__queens_{c}__run_stats_df.csv'))
    rhc_df = rhc_df[~rhc_df.Iteration.isin([1024,2048])]
    SA_df = pd.read_csv(os.path.join(INPUT_DIRECTORY, f'queens_{c}', f'sa__queens_{c}__run_stats_df.csv'))
    SA_df = SA_df[~SA_df.Iteration.isin([1024,2048])]
    GA_df = pd.read_csv(os.path.join(INPUT_DIRECTORY, f'queens_{c}', f'ga__queens_{c}__run_stats_df.csv'))
    GA_df = GA_df[~GA_df.Iteration.isin([1024,2048])]
    mimic_df = pd.read_csv(os.path.join(INPUT_DIRECTORY, f'queens_{c}', f'mimic__queens_{c}__run_stats_df.csv'))
    mimic_df = mimic_df[~mimic_df.Iteration.isin([1024,2048])]

    queens_ATI_rhc.append(np.concatenate([np.repeat(A,B) \
                    for A,B in zip(np.array([ds['Time'] \
                      for ds in np.split(rhc_df, len(rhc_df) / len(X))]).mean(0), dX)]).mean())
    queens_ATI_sa.append(np.concatenate([np.repeat(A,B) \
                    for A,B in zip(np.array([ds['Time'] \
                      for ds in np.split(SA_df, len(SA_df) / len(X))]).mean(0), dX)]).mean())
    queens_ATI_ga.append(np.concatenate([np.repeat(A,B) \
                    for A,B in zip(np.array([ds['Time'] \
                      for ds in np.split(GA_df, len(GA_df) / len(X))]).mean(0), dX)]).mean())
    queens_ATI_mimic.append(np.concatenate([np.repeat(A,B) \
                    for A,B in zip(np.array([ds['Time'] \
                      for ds in np.split(mimic_df, len(mimic_df) / len(X))]).mean(0), dX)]).mean())

    avg_fit_rhc = np.array([ds['Fitness'] for ds in np.split(rhc_df, len(rhc_df) / len(X))]).mean(0)
    avg_fit_rhc = avg_fit_rhc.max() - avg_fit_rhc  # Invert
    avg_fit_rhc = avg_fit_rhc.max() - avg_fit_rhc  # Invert
    queens_FPI_ax.plot(X, avg_fit_rhc, linestyle=lines[i], color='red', label=f'RHC Queens_N={c}')
    avg_fit_SA = np.array([ds['Fitness'] for ds in np.split(SA_df, len(SA_df) / len(X))]).mean(0)
    avg_fit_SA = avg_fit_SA.max() - avg_fit_SA  # Invert
    avg_fit_SA = avg_fit_SA.max() - avg_fit_SA  # Invert
    queens_FPI_ax.plot(X, avg_fit_SA, linestyle=lines[i], color='blue', label=f'SA Queens_N={c}')
    avg_fit_GA = np.array([ds['Fitness'] for ds in np.split(GA_df, len(GA_df) / len(X))]).mean(0)
    avg_fit_GA = avg_fit_GA.max() - avg_fit_GA  # Invert
    avg_fit_GA = avg_fit_GA.max() - avg_fit_GA  # Invert
    queens_FPI_ax.plot(X, avg_fit_GA, linestyle=lines[i], color='green', label=f'GA Queens_N={c}')
    avg_fit_mimic = np.array([ds['Fitness'] for ds in np.split(mimic_df, len(mimic_df) / len(X))]).mean(0)
    avg_fit_mimic = avg_fit_mimic.max() - avg_fit_mimic  # Invert
    avg_fit_mimic = avg_fit_mimic.max() - avg_fit_mimic  # Invert
    queens_FPI_ax.plot(X, avg_fit_mimic, linestyle=lines[i], color='orange', label=f'MIMIC Queens_N={c}')

    queens_FPI_ax.set_title(f'Average Fitness for N-Queens, N = {c}')
    queens_FPI_ax.set_xlabel('Iterations')
    queens_FPI_ax.set_ylabel('Fitness')
    queens_FPI_ax.set_xscale('log')
    queens_FPI_ax.legend()
    queens_FPI_Fig.set_size_inches((15, 10))
    plt.savefig(os.path.join(OUTPUT_DIRECTORY,f'queens_{c}_FPI.png'), dpi=400, format='png')
    plt.close(queens_FPI_Fig)

queens_ATI_Fig, queens_ATI_ax = plt.subplots()
queens_ATI_ax.set_title('Analog of Computational Load for N-Queens')
queens_ATI_ax.set_xlabel('Number of Graph Nodes')
queens_ATI_ax.set_ylabel('Average Time / Iteration')
queens_ATI_ax.plot(qN, queens_ATI_rhc, 'o-', color='red', label='RHC')
queens_ATI_ax.plot(qN, queens_ATI_sa, 'o-', color='blue', label='SA')
queens_ATI_ax.plot(qN, queens_ATI_ga, 'o-', color='green', label='GA')
queens_ATI_ax.plot(qN, queens_ATI_mimic, 'o-', color='orange', label='MIMIC')
queens_ATI_ax.set_yscale('log')
queens_ATI_ax.legend()
queens_ATI_Fig.set_size_inches((10, 10))
queens_ATI_Fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIRECTORY,'queens_ATI.png'), dpi=400, format='png')
plt.close(queens_ATI_Fig)
