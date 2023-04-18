import numpy as np

from crp import CRP
import utils

from copy import deepcopy


class View(CRP):
    def __init__(self, dims, alpha=None, rng=None):
        super().__init__(alpha, rng)
        self.dims = dims  # set of dimensions

    def transition_clusters(self):
        random_order_rows = self.rng.permutation(self.N)

        for row in random_order_rows:
            cluster_i = self.cluster_of(row)

            # calc priors:
            priors = self.calc_transition_log_probs(cluster_i)

            # calc likes:
            log_liks = np.zeros(len(self.elements) + 1)

            for dim in self.dims:
                log_liks += self.posterior_pred_dim(dim, row)

            probs = utils.exp_normalize(priors + log_liks)
            choice = self.rng.choice(len(probs), p=probs)

            self.move_row(row, cluster_i, choice)

    def transition_hypers(self):
        for dim in self.dims:
            dim.transition_hypers()

    def posterior_pred_dim(self, dim, row):
        """posterior log predictive of observing a row in every existing cluster for this dimension"""

        log_liks = np.zeros(len(self.elements) + 1)

        for cluster_i, cluster in enumerate(self.elements):
            x = dim.data[row]

            if row in cluster:
                dim.remove_from_cluster(x, cluster_i, keep=True)
                log_liks[cluster_i] += dim.posterior_pred(x, cluster_i)
                dim.add_to_cluster(x, cluster_i)
            else:
                log_liks[cluster_i] += dim.posterior_pred(x, cluster_i)

        log_liks[len(self.elements)] = dim.posterior_pred(x, len(self.elements))

        return log_liks

    def marginal_lik_dim(self, dim):
        """marginal likelihood of observing a dim under this view"""

        # if dim not inside of this view, construct hypothetical dim clustered under this view
        if self.dims is None or dim not in self.dims:
            dim = deepcopy(dim)
            dim.set_suff_stats(self.elements)

        log_prob = sum([dim.marginal_lik(cluster_i) for cluster_i in range(self.K)])

        return log_prob

    def move_row(self, row, orig, dest):
        """move a row from one cluster to another"""
        in_different_row = super().add_obj(row, dest)

        if in_different_row:
            for dim in self.dims:
                dim.add_to_cluster(dim.data[row], dest)
                dim.remove_from_cluster(dim.data[row], orig)

            super().remove_obj(row, orig)

    def get_dim_indeces(self):
        return [dim.index for dim in self.dims]

    def _transition_params(self):
        """only for collapsed dtypes, not implemented"""
        raise NotImplementedError
        # for dim in self.dims:
        #    dim.transition_params(self.elements)
