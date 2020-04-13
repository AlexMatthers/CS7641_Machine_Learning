import numpy as np
import pandas as pd
import time
import solvers
import gym
from toh_env import TohEnv
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pathlib
import pickle

SEED = 10
OUTFILE = r'Data'
pathlib.Path(OUTFILE).mkdir(parents=True, exist_ok=True)

def run_solver(solver, MAX_STEPS=5000):
    stats = {}
    t = time.perf_counter()
    stepC = 0
    bestP = None
    bestR = -np.inf
    bestV = None
    reward_L = []
    delta_L = []
    steps_L = []
    step_time_L = []
    mean_V = []
    done = False
    solver.reset()

    while not done and stepC < MAX_STEPS:
        P, V, steps, step_time, R, delta, done = solver.step()
        if R > bestR:
            bestR = R
            bestP = P
            bestV = V

        reward_L.append(R)
        delta_L.append(delta)
        steps_L.append(steps)
        step_time_L.append(step_time)
        mean_V.append(np.mean(V))
        stepC += 1
        if stepC % 50 == 0:
            print(stepC)
            print(step_time)
    print('done')
    elapsedT = time.perf_counter() - t
    stats['bestR'] = bestR
    stats['bestP'] = bestP
    stats['bestV'] = bestV
    stats['elapsedT'] = elapsedT
    stats['reward'] = reward_L
    stats['delta'] = delta_L
    stats['steps'] = steps_L
    stats['step_time'] = step_time_L
    stats['mean_V'] = mean_V
    stats['score'] = score_policy(solver, bestP)
    return stats

def score_policy(solver, policy, iters=100):
    pScore = []
    for i in range(iters):
        pScore.append(np.sum(solver.run_policy(policy)))
    return np.mean(pScore)

def run_experiment(solver, env, name):
    data = {}
    env.reset()
    gammas = [0.7, 0.8, 0.9, 0.99, 0.999]
    if solver.name in ['VI', 'PI']:
        for g in gammas:
            print(f'Gamma={g}')
            v = solver(env=env, gamma=g)
            stats = run_solver(v)
            data[str(g)] = stats
    else:
        eps = [100000]
        steps = [350]
        alpha = [0.001, 0.1]
        epsilon = [0.1, 0.7]
        epsilon_decay = [0.0005, 0.00001]

        param_grid = {'gamma':gammas[::2],
                      'max_episodes':eps,
                      'max_steps':steps,
                      'alpha':alpha,
                      'epsilon':epsilon,
                      'epsilon_decay':epsilon_decay}
        grid = ParameterGrid(param_grid)

        for i, param in enumerate(grid):
            print(f'Set: {i}, Params: {param}')
            v = solver(env=env,
                       m_eps=param['max_episodes'],
                       m_steps=param['max_steps'],
                       gamma=param['gamma'],
                       alpha=param['alpha'],
                       epsilon=param['epsilon'],
                       eps_decay=param['epsilon_decay'],
                       initQ=1.0)
            stats = run_solver(v, MAX_STEPS=500000)
            stats['params'] = param
            data[i] = stats

    return v, pd.DataFrame.from_dict(data)

def process(v, data, outfile, envName, solveName):
    bestIdx = data.T.index[np.argmax(data.T.score)]
    V = data[bestIdx].T.bestV
    n = int(V.size**0.5)
    # if solveName in ['PI', 'VI'] else data[data.T.score.argmax()].params
    if envName == 'FrozenLake':
        arrows = np.array(['\u2190', '\u2193', '\u2192', '\u2191'])
        FLmap = np.zeros((n,n,4))
        FLmap[v.env.desc==b'S'] = cm.Paired(3)
        FLmap[v.env.desc==b'G'] = cm.Paired(5)
        FLmap[v.env.desc==b'F'] = cm.Paired(0)
        FLmap[v.env.desc==b'H'] = cm.Paired(1)
        a = arrows[data[bestIdx].T.bestP.argmax(1)]

        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[0].set_title(f'Policy Map for FrozenLake with {solveName}')
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[1].set_title(f'Value Map for FrozenLake with {solveName}')
        im1 = ax[0].imshow(FLmap)
        im2 = ax[1].imshow(V.reshape(n, n))
        kw = dict(horizontalalignment="center", verticalalignment="center")
        textcolors = ['black', 'white']
        for i in range(n):
            for j in range(n):
                kw.update(color=textcolors[int(im2.norm(np.round(V.reshape(n,n),2)[i,j]) < V.max()/2)])
                _ = im1.axes.text(j, i, a.reshape(n,n)[i,j], **dict(horizontalalignment="center", verticalalignment="center"))
                _ = im2.axes.text(j, i, np.round(V.reshape(n,n),2)[i,j], **kw)

        fig.tight_layout()
        plt.savefig(os.path.join(outfile, f'{solveName}_{envName}_visGraphs.png'), dpi=400, format='png')
        plt.close(fig)
    else:
        pass

    plt.figure(figsize=(10,10))
    plt.scatter(data.T.index, data.T.score)
    plt.title('Accuracy Score by parameter')
    if solveName=='QL':
        plt.xlabel('Parameter Combo Index')
    else:
        plt.xlabel('Gamma')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(outfile, f'{solveName}_{envName}_accuracy.png'), dpi=400, format='png')
    plt.close()

    plt.figure(figsize=(10,10))
    [plt.plot(range(len(data[i].T.delta)), data[i].T.delta, label=i) for i in data.columns]
    plt.title('Delta Convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Delta (max. change in Value function)')
    plt.legend()
    plt.savefig(os.path.join(outfile, f'{solveName}_{envName}_deltas.png'), dpi=400, format='png')
    plt.close()

def comparePolicy(data1, data2):
    bestIdx1 = data1.T.index[np.argmax(data1.T.score)]
    bestIdx2 = data2.T.index[np.argmax(data2.T.score)]
    comp = data1[bestIdx1].T.bestP == data2[bestIdx2].T.bestP
    return comp.sum() / comp.size


if __name__ == '__main__':
    # env3_slip = gym.make('FrozenLake-v0', desc=LAKEMAP_20_20)
    env1_slip = gym.make('FrozenLake-v0', map_name="8x8")
    # env1_noslip = gym.make('FrozenLake-v0', map_name="8x8", is_slippery=False)
    env2_noslip = TohEnv(3, 6, 0, stepReward=-0.05, invalidReward=-1.0)
    # env2_slip = TohEnv(3, 5, 0.2, stepReward=-0.005, invalidReward=-0.25)
    VI = solvers.ValueIterationSolver
    PI = solvers.PolicyIterationSolver
    QL = solvers.QLearnerSolver
    ## VI_FLns, VI_FLns_data = run_experiment(VI, env1_noslip, '')  # < 1 min
    ## PI_FLns, PI_FLns_data = run_experiment(PI, env1_noslip, '')  # ~5 min
    ## QL_FLns, QL_FLns_data = run_experiment(QL, env1_noslip, '')
    ## VI_ToHs, VI_ToHs_data = run_experiment(VI, env2_slip, '')
    ## PI_ToHs, PI_ToHs_data = run_experiment(PI, env2_slip, '')
    ## QL_ToHs, QL_ToHs_data = run_experiment(QL, env2_slip, '')

    print('VI_FLs')
    VI_FLs, VI_FLs_data = run_experiment(VI, env1_slip, 'FL')
    process(VI_FLs_data, OUTFILE, 'FrozenLake', 'VI')
    pickle.dump([VI_FLs, VI_FLs_data], open('VI_FLs.p', 'wb+'))
    print('VI_ToH')
    VI_ToHns, VI_ToHns_data = run_experiment(VI, env2_noslip, 'ToH')  # < 1 min
    process(VI_ToHns_data, OUTFILE, 'ToH', 'VI')
    pickle.dump([VI_ToHns, VI_ToHns_data], open('VI_ToHns.p', 'wb+'))

    print('PI_FLs')
    PI_FLs, PI_FLs_data = run_experiment(PI, env1_slip, 'FL')
    process(PI_FLs_data, OUTFILE, 'FrozenLake', 'PI')
    pickle.dump([PI_FLs, PI_FLs_data], open('PI_FLs.p', 'wb+'))
    print('PI_ToH')
    PI_ToHns, PI_ToHns_data = run_experiment(PI, env2_noslip, 'ToH')  # 35 min
    process(PI_ToHns_data, OUTFILE, 'ToH', 'PI')
    pickle.dump([PI_ToHns, PI_ToHns_data], open('PI_ToHns.p', 'wb+'))
    print(f'Solution similarity: {comparePolicy(PI_FLs_data, VI_FLs_data)*100}% between PI and VI for Frozen Lake')
    print(f'Solution similarity: {comparePolicy(PI_ToHns_data, VI_ToHns_data)*100}% between PI and VI for Towers of Hanoi')
    print('QL_FLs')
    QL_FLs, QL_FLs_data = run_experiment(QL, env1_slip, 'FL')
    process(QL_FLs_data, OUTFILE, 'FrozenLake', 'QL')
    pickle.dump([QL_FLs, QL_FLs_data], open('QL_FLs.p', 'wb+'))
    print('QL_ToH')
    QL_ToHns, QL_ToHns_data = run_experiment(QL, env2_noslip, 'ToH')
    process(QL_ToHns_data, OUTFILE, 'ToH', 'QL')
    pickle.dump([QL_ToHns, QL_ToHns_data], open('QL_ToHns.p', 'wb+'))
    print(f'Solution similarity: {comparePolicy(PI_FLs_data, QL_FLs_data)*100}% between PI and QL for Frozen Lake')
    print(f'Solution similarity: {comparePolicy(QL_FLs_data, VI_FLs_data)*100}% between QL and VI for Frozen Lake')
    print(f'Solution similarity: {comparePolicy(PI_ToHns_data, QL_ToHns_data)*100}% between PI and QL for Towers of Hanoi')
    print(f'Solution similarity: {comparePolicy(QL_ToHns_data, VI_ToHns_data)*100}% between QL and VI for Towers of Hanoi')


    ## Delta is change in state value aka TD delta
