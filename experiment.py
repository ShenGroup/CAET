
# CAET algorithm
# Special case for regret minimization in three-armed bandit case during ranking identification case
# The configuration part of the parameter is in the last of this file


import numpy as np
import matplotlib.pyplot as plt
import random
import math

class GaussianArm():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.mean_return = self.mu

    def draw(self):
        return random.gauss(self.mu, self.sigma)


"""
class Policy
"""


class CAET_3():
    # just for three arm bandit and regret minimization case
    def __init__(self, confidence, r):
        self.delta = confidence
        self.alpha = 1 - np.power(np.log(1 / confidence), -r)
        self.thre = 0.1 * np.power(np.log(1 / confidence), -r / 4)
        self.npulls = np.zeros(3, dtype=np.int32)
        self.sample_mean = np.zeros(3)
        self.cost_mean = np.zeros(3)
        self.track_estimates = np.zeros(3)
        self.t = 0

    def cut_off(self, x): # truncation function D_\delta
        #if x < self.delta ** 2:
        if x < self.thre:
            return 0

        return x

    def projection(self, u, time): # calculate projection
        epslion = np.power((9 + time), -0.5) / 2
        num = 0
        u_epsilon = np.zeros(3)
        amin = np.argmin(u)
        amax = np.argmax(u)
        amid = 3 - amax - amin

        for i in range(3):
            if u[i] <= epslion:
                num += 1
        if num == 0:
            return u
        elif num == 1:
            u_epsilon[(amin - 1) % 3] = u[(amin - 1) % 3] - (epslion - u[amin]) / 2
            u_epsilon[(amin - 2) % 3] = u[(amin - 2) % 3] - (epslion - u[amin]) / 2
            u_epsilon[amin] = epslion
            aamin = np.argmin(u_epsilon)
            aamax = np.argmax(u_epsilon)
            if u_epsilon[aamin] <= epslion:
                u_epsilon[aamin] = epslion
                u_epsilon[aamax] = 1 - 2 * epslion
            return u_epsilon
        elif num == 2:
            u_epsilon[amin] = epslion
            u_epsilon[amid] = epslion
            u_epsilon[amax] = 1 - 2 * epslion
            return u_epsilon
        else:
            return u




    def select_arm(self):
        """
        If an arm has never been selected, its UCB is viewed as infinity
        """
        for arm in range(3):
            if self.npulls[arm] == 0:
                return arm
        #ucb = [0.0 for arm in range(3)]
        ucb = np.zeros(3)
        for arm in range(3):
            ucb[arm] = self.track_estimates[arm] - self.npulls[arm]
        return np.argmax(ucb)

    def update(self, chosen_arm, reward):
        self.npulls[chosen_arm] += 1
        n = self.npulls[chosen_arm]
        value = self.sample_mean[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.sample_mean[chosen_arm] = new_value
        best_arm = np.argmax(self.sample_mean)
        self.cost_mean[best_arm] = 0
        for arm in range(3):
            if arm != best_arm:
                self.cost_mean[arm] = self.sample_mean[best_arm] - self.sample_mean[arm]
                self.cost_mean[arm] = self.cut_off(self.cost_mean[arm])
        self.culculate()
        self.t += 1

    def culculate(self):
        ident = np.zeros(3)
        omega = np.zeros(3)
        u = np.zeros(3)
        u_alpha = np.zeros(3)
        secopt = 0
        num = 0
        d_2 = 0.0
        d_3 = 0.0
        for arm in range(3):
            if self.cost_mean[arm] == 0:
                num += 1
        for arm in range(3):
            if self.cost_mean[arm] == 0:
                ident[arm] = self.alpha / num
        opt = np.argmax(self.cost_mean)  # arm with maximum sub-optimal gap
        if num == 1:
             # arm with second maximum sub-optimal gap
            for arm in range(3):
                if (arm != opt) and (self.cost_mean[arm] != 0):
                    secopt = arm
            d_2 = self.cost_mean[secopt]
            d_3 = self.cost_mean[opt]
            if d_3 / d_2 <= (3 + 5 ** 0.5) / 2:
                omega[secopt] = d_2 ** 0.5 / (d_2 ** 0.5 + d_2 ** 0.5)
                omega[opt] = d_3 ** 0.5 / (d_2 ** 0.5 + d_2 ** 0.5)

            else:
                omega[secopt] = (d_3 - 2 * d_2 )/ (d_3 - d_2)
                omega[opt] = d_2 / (d_3 - d_2)

            u[opt] = (omega[opt] / d_3) / (omega[opt] / d_3 + omega[secopt] / d_2)
            u[secopt] = (omega[secopt] / d_2) / (omega[opt] / d_3 + omega[secopt] / d_2)
        else:
            omega[opt] = 1
            u[opt] = 1

        u_alpha = ident + (1 - self.alpha) * u
        # print(self.cost_mean)
        # print(secopt, opt)
        # print(np.power((9 + self.t), -0.5) / 2)
        # print(u_alpha)
        # print(self.projection(u_alpha, self.t))
        self.track_estimates += self.projection(u_alpha, self.t)

    def z_a_b(self, a, b):
        hat_mu_ab = self.npulls[a] * self.sample_mean[a] / (self.npulls[a] + self.npulls[b]) + self.npulls[b] * self.sample_mean[b] / (self.npulls[a] + self.npulls[b])
        d_a = ((self.sample_mean[a] - hat_mu_ab) ** 2) / 2
        d_b = ((self.sample_mean[b] - hat_mu_ab) ** 2) / 2
        return self.npulls[a] * d_a + self.npulls[b] * d_b





    def reset(self):
        self.npulls = np.zeros(3, dtype=np.int32)
        self.sample_mean = np.zeros(3)
        self.cost_mean = np.zeros(3)
        self.track_estimates = np.zeros(3)
        self.t = 0

    def get_name(self):
        return "CAET"



class Environment():
    def __init__(self, bandits, agents, theta):
        self.bandits = bandits
        self.agents = agents
        self.results = None
        self.ranking = None
        self.theta = theta
        self.K = len(self.bandits)

    def reset(self):
        self.ranking = None
        self.agents.reset()
    def regret(self):# calculate the regret
        mu = np.zeros(3)
        for i in range(3):
            mu[i] = self.bandits[i].mean_return
        opt = np.argmax(mu)
        min = np.argmin(mu)
        sec = 3 - opt - min
        d_2 = mu[opt] - mu[sec]
        d_3 = mu[opt] - mu[min]
        if d_3 / d_2 <= (3 + 5 ** 0.5) / 2:
            return (2 / (d_2 ** 0.5 - d_3 ** 0.5) ** 2) * np.log(1 / self.agents.delta)
        else:
            return 2 * (d_3 - d_2 ) / (d_2 * d_3 - 2 * d_2 * d_2) * np.log(1 / self.agents.delta)

    def run(self, horizon=10000, experiments=1):
        results = np.zeros((self.M, experiments, horizon))
        for m in range(self.M):
            agent = self.agents[m]
            for i in range(experiments):
                self.reset()
                for t in range(horizon):
                    action = agent.select_arm()
                    reward = self.bandits[action].draw()
                    results[m][i][t] = reward
                    agent.update(action, reward)

        self.results = results


    def run_stop(self): #  stopping rule
        ranking = np.zeros(3)
        self.reset()
        agenta = self.agents
        for t in range(3):
            action = self.agents.select_arm()
            reward = self.bandits[action].draw()
            self.agents.update(action, reward)
        while min(agenta.z_a_b(0, 1), agenta.z_a_b(1, 2), agenta.z_a_b(0, 2)) < np.log((1000000000000 * agenta.t ** self.theta) / agenta.delta):
            action = agenta.select_arm()
            reward = self.bandits[action].draw()
            agenta.update(action, reward)
        ranking[0] = np.argmax(agenta.sample_mean) + 1
        ranking[2] = np.argmin(agenta.sample_mean) + 1
        ranking[1] = 6 - ranking[0] - ranking[2]

        self.ranking = ranking

    def show_results(self): # output regret, ranking and other information
        if self.ranking is None:
            print("No results yet.")
            return -1
        reward = 0
        for i in range(3):
            reward += self.agents.npulls[i] * self.bandits[i].mean_return
        regret = self.agents.t * max(self.bandits[0].mean_return, self.bandits[1].mean_return, self.bandits[2].mean_return) - reward
        print("confidence", self.agents.delta) # These are the data which running this file will output
        print("total time", self.agents.t)
        print("bandits means", self.bandits[0].mean_return, self.bandits[1].mean_return, self.bandits[2].mean_return)
        print("pulling time of each arm", self.agents.npulls)
        print("track estimites", self.agents.track_estimates)
        print("Alpha", self.agents.alpha)
        print("Regret", regret)
        print("regret lower bound", self.regret())
        print("regret upper bound", self.regret() * self.theta)
        print("Ranking", self.ranking)


# The following is parameter configuration part.

Delta = 1e-70 # confidence
R = 0.4 # This parameter is r in the CAET algorithm
mu_1 = 1.4 # These mu_1, mu_2 and mu_3 are the mean of the three arms
mu_2 = 0.8
mu_3 = 0.3

bandits = [GaussianArm(mu_1, 1), GaussianArm(mu_2, 1), GaussianArm(mu_3, 1)]

agent = CAET_3(Delta, R)


MAB = Environment(bandits, agents=agent, theta=1.2)# The theta represents the theta in the paper which is the scale of the optimal regret
MAB.run_stop()
MAB.show_results()
