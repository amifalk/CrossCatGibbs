import numpy as np
from state import State

from joblib import Parallel, delayed

from copy import deepcopy
import pickle
from fastprogress.fastprogress import progress_bar

import seaborn as sns
import matplotlib.pyplot as plt


class CrossCat:
    def __init__(self, data, n_chains=1, dists=None, alpha=None, views=None, cats=None):
        self.data = data
        self.n_chains = n_chains

        self.dists = dists
        if self.dists is None:
            self.dists = self.infer_dists()

        self.alpha = alpha
        self.views = views
        self.cats = cats

        self.curr_states = [
            State(self.data, self.dists, self.alpha, self.views, self.cats)
            for i in range(n_chains)
        ]

        self.states = [deepcopy(self.curr_states)]
        self.liks = [[state.log_lik() for state in self.curr_states]]

    def sample(self, num_iter=10, sample_every=1):
        for i in progress_bar(range(num_iter)):
            if self.n_chains > 1:
                self.curr_states = Parallel(n_jobs=-1)(
                    delayed(_sample)(state, sample_every) for state in self.curr_states
                )
            else:
                self.curr_states = [_sample(self.curr_states[0], sample_every)]

            self.states.append(self.curr_states)
            self.liks.append([state.log_lik() for state in self.curr_states])

    def best_fits(self):
        liks = np.array(self.liks)
        best_iters = np.argmax(liks, axis=0)  # best iteration for each chain
        best_fits = [
            deepcopy(self.states[iter][chain_i])
            for chain_i, iter in enumerate(best_iters)
        ]

        return best_fits

    def convergence_plot(self):
        liks = np.array(self.liks)
        sns.lineplot(liks)
        plt.title(f"Model Log Likelihood/Time")
        plt.xlabel("Iterations")
        plt.ylabel("Log Likelihood")
        plt.show()

    def infer_dists(self):
        """guess the distributions that each column belongs to"""
        dists = []

        for col in self.data.T:
            if np.all((col == 1) | (col == 0)):
                dists.append("bern")
            else:
                dists.append("normal")

        return dists

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            crosscat_obj = pickle.load(file)

        return crosscat_obj

    def __repr__(self):
        s = f"Chains: {self.n_chains}, Samples: {len(self.states)}, Data: {self.data.shape}"

        return s

def _sample(state, n_transitions=1):
    # n_transitions = progress_bar(range(n_transitions))

    for i in range(n_transitions):
        state.transition()

    return deepcopy(state)
