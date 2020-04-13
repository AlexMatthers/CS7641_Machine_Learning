import numpy as np
import time

def actionVals(env, gamma, state, V):
    A = np.zeros(env.nA)
    for action in range(env.nA):
        for T, nextState, R, _ in env.P[state][action]:
            A[action] += T * (R + gamma * V[nextState])
    return A

class ValueIterationSolver():

    name = 'VI'
    def __init__(self, env, gamma=0.9, theta=0.00001):

        self.env = env
        self.theta = theta
        self.gamma = gamma

        self.V = np.zeros(self.env.nS)
        self.P = np.zeros((self.env.nS, self.env.nA))
        self.steps = 0
        self.stepTime = 0
        self.deltaLast = theta

    def step(self):
        stime = time.perf_counter()
        delta = [0]
        R = 0

        for state in range(self.env.nS):
            A = actionVals(self.env, self.gamma, state, self.V)
            bestAVal = np.max(A)
            delta.append(np.max([delta[-1], np.abs(bestAVal - self.V[state])]))
            R += bestAVal
            self.V[state] = bestAVal

        self.stepTime = time.perf_counter() - stime
        self.deltaLast = delta[-1]
        self.steps += 1

        for state in range(self.env.nS):
            A = actionVals(self.env, self.gamma, state, self.V)
            bestAction = np.argmax(A)
            self.P[state] = [0] * self.env.nA
            self.P[state, bestAction] = 1.0

        return self.P, self.V, self.steps, self.stepTime, R, self.deltaLast, (self.deltaLast < self.theta)

    def reset(self):
        self.V = np.zeros(self.env.nS)
        self.P = np.zeros((self.env.nS, self.env.nA))
        self.steps = 0
        self.stepTime = 0
        self.deltaLast = self.theta

    def run_policy(self, policy, render=False, initState=None):
        P = policy.argmax(1)
        rewards = []
        if initState is None:
            state = self.env.reset()
        else:
            state = self.env.inverse_mapping[initState]
            self.env.s = self.env.inverse_mapping[initState]
        done = False
        steps = 0
        while not done and steps < 1000:
            if render:
                self.env.render()

            action = P[state]
            state, R, done, _ = self.env.step(action)
            rewards.append(R)
            steps += 1

        if render:
            self.env.render()

        return rewards

class PolicyIterationSolver():

    name = 'PI'
    def __init__(self, env, gamma=0.9, maxSteps=1000):
        self.env = env
        self.gamma = gamma
        self.maxSteps = maxSteps

        self.P = np.ones((self.env.nS, self.env.nA)) / self.env.nA
        self.steps = 0
        self.stepTime = 0
        self.deltaLast = 0
        self.stable = False
        # self.T = np.zeros((env.nS, env.nA, env.nS))
        # self.R = np.zeros(env.nS)
        # for s in range(self.env.nS):
        #     for a in range(self.env.nA):
        #         p, ns, r = np.vstack(self.env.P[s][a])[:,:-1].T
        #         self.T[s, a][ns.astype(int)] = p
        #         self.R[ns.astype(int)] = np.where(self.R[ns.astype(int)] == 0, r, self.R[ns.astype(int)])

    def eval_P(self):
        V = np.zeros(self.env.nS)
        steps = 0
        # bAct = self.P.argmax(1)
        while self.maxSteps is None or steps < self.maxSteps:
            delta = 0
            # expV = self.R + self.gamma*np.array([self.T[:, bAct, s]*V[s] for s in range(self.env.nS)]).sum((0,2))
            # expV = expV / expV.max()
            # delta = np.abs(expV-V).max()
            # V = expV
            for state in range(self.env.nS):
                expectedV = 0
                for action, actionT in enumerate(self.P[state]):
                    for T, nextState, R, _ in self.env.P[state][action]:
                        expectedV += actionT * T * (R + self.gamma * V[nextState])
                delta = np.max([delta, np.abs(expectedV - V[state])])
                V[state] = expectedV

            steps += 1
            if delta < 0.0001:
                break

        return V

    def step(self):
        stime = time.perf_counter()
        V = self.eval_P()
        self.stable = True
        delta = [0]
        R = 0

        for state in range(self.env.nS):
            policyAction = np.argmax(self.P)
            A = actionVals(self.env, self.gamma, state, V)
            bestAction = np.argmax(A)
            bestAVal = np.max(A)
            delta.append(np.max([delta[-1], np.abs(bestAVal - V[state])]))
            R += bestAVal

            if policyAction != bestAction:
                self.stable = False

            self.P[state] = [0.0] * self.env.nA
            self.P[state, bestAction] = 1.0
        if delta[-1] >= self.deltaLast and self.steps > 10:
            self.stable = True

        self.steps += 1
        self.stepTime = time.perf_counter() - stime
        self.deltaLast = delta[-1]

        return self.P, V, self.steps, self.stepTime, R, self.deltaLast, self.stable

    def reset(self):
        self.P = np.ones((self.env.nS, self.env.nA)) / self.env.nA
        self.steps = 0
        self.stepTime = 0
        self.deltaLast = 0
        self.stable = False

    def run_policy(self, policy, render=False, initState=None):
        P = policy.argmax(1)
        rewards = []
        if initState is None:
            state = self.env.reset()
        else:
            state = self.env.inverse_mapping[initState]
            self.env.s = self.env.inverse_mapping[initState]
        done = False
        steps = 0
        while not done and steps < 1000:
            if render:
                self.env.render()

            action = P[state]
            state, R, done, _ = self.env.step(action)
            rewards.append(R)
            steps += 1

        if render:
            self.env.render()

        return rewards

class QLearnerSolver():

    name = 'QL'

    def __init__(self, env, m_eps, m_steps, gamma=0.9, alpha=1.0, epsilon=0.1, eps_decay=0.001, theta=0.00001, initQ=0.0):

        self.env = env
        self.maxEps = m_eps
        self.maxSteps = m_steps
        self.gamma = gamma
        self.alpha = alpha
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = eps_decay
        self.theta = theta
        self.initQ = initQ

        self.actions = self.env.action_space.n
        self.Q = np.full((self.env.observation_space.n, self.actions), self.initQ)
        self.steps = 0
        self.stepTime = 0
        self.deltaLast = 0
        self.deltaMark = 0

    def step(self):
        stime = time.perf_counter()

        state = self.env.reset()

        totR = 0.0
        for t in range(self.maxSteps + 1):
            actionT = self.policy_values(state)
            action = np.random.choice(np.arange(self.env.nA), p=actionT)
            nextState, R, done, _ = self.env.step(action)

            nextAction = np.argmax(self.Q[nextState])
            target = R + self.gamma * self.Q[nextState, nextAction]
            delta = target - self.Q[state, action]
            self.Q[state, action] += self.alpha * delta

            self.epsilon -= self.epsilon * self.epsilon_decay

            totR += R
            self.deltaLast = np.abs(delta)

            if done:
                break

            state = nextState
        if self.deltaLast < self.theta:
            self.deltaMark += 1
        else:
            self.deltaMark = 0

        self.stepTime = time.perf_counter() - stime
        self.steps +=1
        try:
            Rfin = totR / t
        except:
            Rfin = totR
        return self.policy(), self.values(), self.steps, self.stepTime, Rfin, self.deltaLast, self.converged()


    def policy_values(self, state):
        P = np.ones(self.actions) * self.epsilon / self.actions
        bestAction = np.argmax(self.Q[state])
        P[bestAction] += (1.0 - self.epsilon)
        return P

    def policy(self):
        P = np.zeros((self.env.nS, self.env.nA))
        for state in range(self.env.nS):
            bestAction = np.argmax(self.Q[state])
            P[state, bestAction] = 1.0
        return P

    def values(self):
        V = np.zeros(self.env.nS)
        for state in range(self.env.nS):
            V[state] = np.max(self.Q[state])
        return V

    def reset(self):
        self.Q = np.full((self.env.observation_space.n, self.actions), self.initQ)
        self.steps = 0
        self.stepTime = 0
        self.deltaLast = 0
        self.initial_epsilon = self.epsilon
        self.deltaMark = 0

    def converged(self):
        return (self.steps >= 20 and self.deltaMark >= 10) or self.steps >= self.maxEps

    def run_policy(self, policy, render=False, initState=None):
        P = policy.argmax(1)
        rewards = []
        if initState is None:
            state = self.env.reset()
        else:
            state = self.env.inverse_mapping[initState]
            self.env.s = self.env.inverse_mapping[initState]
        done = False
        steps = 0
        while not done and steps < 1000:
            if render:
                self.env.render()

            action = P[state]
            state, R, done, _ = self.env.step(action)
            rewards.append(R)
            steps += 1

        if render:
            self.env.render()

        return rewards
