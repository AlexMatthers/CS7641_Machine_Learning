import mlrose_hiive as mlr
import numpy as np
import os
import sys
import multiprocessing as mp
import types
import time
from pathlib import Path

RANDOM_STATE = 10
THREADS = -1
OUTPUT_DIRECTORY = os.path.join(sys.path[0],'Output')
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)

print("Building Problems...")
## Problems
problems_list = []
# 4-color - Custom fitness to maximize

colorGen = mlr.MaxKColorGenerator()
# cN = 25
k = 5
r = 4
# color_Prob = colorGen.generate(RANDOM_STATE, number_of_nodes=cN, max_connections_per_node=r, max_colors=k)
cN = [5, 10, 15, 30, 55, 100]
for i in cN:
    color = colorGen.generate(RANDOM_STATE, number_of_nodes=i, max_connections_per_node=r, max_colors=k)
    problems_list.append((color, f'color_{i}'))

# N-Queens - Custom fitness to maximize
# qN = 8
# queens_Prob = mlr.QueensOpt(length=qN)
def QMaxFit(state):
    fitness_cnt = 0
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            if (state[j] != state[i]) and (state[j] != state[i] + (j - i)) and (state[j] != state[i] - (j - i)):
                fitness_cnt += 1
    return fitness_cnt

QMF_cust = mlr.CustomFitness(QMaxFit)  # max val is len(list(itertools.combinations(range(len(state)), 2)))
qN = [20, 35, 50, 65, 80, 95]
for i in qN:
    problems_list.append((mlr.QueensOpt(length=i, fitness_fn=QMF_cust, maximize=True), f'queens_{i}'))

# Continuous Peaks
# pN = 12

t_pct = 0.15
peaksGen = mlr.ContinuousPeaksGenerator()
pN = [5, 10, 15, 30, 55, 100]
for i in pN:
    problems_list.append((peaksGen.generate(RANDOM_STATE, i, t_pct), f'peaks_{i}'))

print('Building experiments...')
## Algorithms
experiments = []
for prob in problems_list:  # Test if file already exists, if so skip
    problem, expName = prob
    iList = 2**np.arange(10)
    # Random Hill Climbing
    if not os.path.isfile(os.path.join(sys.path[0],'Output', expName, f'rhc__{expName}__curves_df.csv')):
        rhc = mlr.RHCRunner(problem=problem,
                            experiment_name=expName,
                            output_directory=OUTPUT_DIRECTORY,
                            seed=RANDOM_STATE,
                            iteration_list=iList,
                            restart_list=[5,10,20,30],
                            verbose=False)
        experiments.append(rhc)
    # s,c = rhc.run()
    # results.append([s,c])
    # Simulated Annealing
    if not os.path.isfile(os.path.join(sys.path[0],'Output', expName, f'sa__{expName}__curves_df.csv')):
        sa = mlr.SARunner(problem=problem,
                          experiment_name=expName,
                          output_directory=OUTPUT_DIRECTORY,
                          seed=RANDOM_STATE,
                          iteration_list=iList,
                          temperature_list=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
                          decay_list=[mlr.ArithDecay, mlr.ExpDecay, mlr.GeomDecay],
                          verbose=False)
        experiments.append(sa)
    # s,c = sa.run()
    # results.append([s,c])
    # Genetic Algorithm
    if not os.path.isfile(os.path.join(sys.path[0],'Output', expName, f'ga__{expName}__curves_df.csv')):
        ga = mlr.GARunner(problem=problem,
                          experiment_name=expName,
                          output_directory=OUTPUT_DIRECTORY,
                          seed=RANDOM_STATE,
                          iteration_list=iList,
                          population_sizes=[100, 200, 300, 400, 500],
                          mutation_rates=[0.3, 0.4, 0.5, 0.6, 0.7],
                          verbose=False)
        experiments.append(ga)
    # s,c = ga.run()
    # results.append([s,c])
    # MIMIC
    if not os.path.isfile(os.path.join(sys.path[0],'Output', expName, f'mimic__{expName}__curves_df.csv')):
        mc = mlr.MIMICRunner(problem=problem,
                            experiment_name=expName,
                            output_directory=OUTPUT_DIRECTORY,
                            seed=RANDOM_STATE,
                            iteration_list=iList,
                            population_sizes=[50, 125, 200, 275, 350],
                            keep_percent_list=[0.25, 0.5, 0.75],
                            verbose=False)
        experiments.append(mc)
    # s,c = mc.run()
    # results.append([s,c])

# experiments is 6 * 3 * 4 = 72 experiments long....
print(len(experiments))
print(np.array([[exp.dynamic_runner_name() + exp._experiment_name for exp in experiments]]).T)
if input('Continue... >') == 'y':
    pass
else:
    quit()
print('Running all experiments...')

def runUtil(exp):
    s,c = exp.run()
    return (exp.runner_name(), exp._experiment_name, s,c)
# print(experiments)
if __name__ == '__main__':

    pool = mp.Pool(processes=THREADS)
    results = pool.map(runUtil, experiments)
    print('Experiments Done.')
    # resDict = {}
    # for res in results:
    #     alg, exp, stats, curves = res
    #     prob, N = exp.split('_')
    #     resDict[prob] = {N: (alg, stats, curves)}

    # FIGURE OUT PLOTTING???

    # Plot of fitness vs iterations

    # Plot of best fitness (avg fitness?) vs problem complexity (N)

    # Plot of average time per iteration vs problem complexity (N)

    # Plot of total average time vs problem complexity (N)
