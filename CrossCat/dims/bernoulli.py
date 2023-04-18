from .dim import Dim
import numpy as np
from scipy.special import betaln
from RunDEMC import Model, Param, dists
import utils


class Bernoulli(Dim):
    """Dimension object for binary data.

    data ~ Bernoulli(theta)

    Prior:
    theta ~ Beta(s*b, s*(1-b))

    Hyperprior:
    s ~ Exponential(5)
    b ~ Beta(1, 1)
    [s, b]
    
    Sufficient Statistics:
    [K, N]
    
    https://github.com/probcomp/cgpm/blob/master/src/primitives/bernoulli.py
    """
    # distribution of hyperparams (fixed)
    s_dist = dists.exp(5)
    b_dist = dists.beta(1, 1)  

    s_i = 0 # strength index
    b_i = 1 # balance index
    K_i = 0 # K index
    N_i = 1 # N index


    def __init__(self, index, data, suff_stats, hypers=None, rng=None):
        super().__init__(index, data, suff_stats, hypers, rng)

    def add_to_cluster(self, x, cluster_i):
        assert cluster_i <= len(self.suff_stats)

        if cluster_i < len(self.suff_stats):
            self.suff_stats[cluster_i][self.K_i] += x
            self.suff_stats[cluster_i][self.N_i] += 1

        elif cluster_i == len(self.suff_stats):
            self.suff_stats.append([x, 1])

    def remove_from_cluster(self, x, cluster_i, keep=False):
        assert self.suff_stats[cluster_i][self.N_i] != 0

        self.suff_stats[cluster_i][self.K_i] -= x
        self.suff_stats[cluster_i][self.N_i] -= 1

        # only allowed to delete the cluster if given permission
        if not keep and self.suff_stats[cluster_i][self.N_i] == 0:
            del self.suff_stats[cluster_i]

    def gen_hypers(self):
        s = self.s_dist.rvs(size=1, random_state=self.rng)[0]
        b = self.b_dist.rvs(size=1, random_state=self.rng)[0]

        return [s, b]

    def p_hypers(self):
        p_s = self.s_dist.logpdf(x=self.hypers[self.s_i])
        p_b = self.b_dist.logpdf(x=self.hypers[self.b_i])
        return p_s + p_b

    def transition_hypers(self, MAP=False, n_samps=100):
        def log_lik(pop, *args):
            """marginal likelihood for the dimension"""
            
            alpha = pop["s"] * pop["b"]
            beta = pop["s"] * (1.0 - pop["b"])

            liks = 0
            for K, N in self.suff_stats:            
                liks += (betaln(alpha + K, beta + N - K) - betaln(alpha, beta))
                
            return liks

        hypers = [
            Param(name="s", prior=self.s_dist),
            Param(name="b", prior=self.b_dist),
        ]

        mod = Model(
            name="fun",
            params=hypers,
            like_fun=log_lik,
            like_args=None,
            initial_zeros_ok=False,
            use_priors=True,
            verbose=False,
        )

        burnin = 200
        mod(n_samps + burnin, burnin=False)
        
        if MAP:
            self.hypers = utils.MAP_estimates(mod, burnin)
        else:
            self.hypers = utils.posterior_sample(mod, burnin)

    def set_suff_stats(self, clusters):
        # TODO: pretty slow to convert a set into a list ...
        self.suff_stats = [
            [np.sum(self.data[list(cluster)]), len(cluster)] for cluster in clusters
        ]
        # [K, N]

    def marginal_lik(self, cluster_i):
        """P(X|cluster with N items, K successes)
        https://gregorygundersen.com/blog/2020/08/19/bernoulli-beta/"""
        alpha, beta = self.get_reparam_hypers()

        K, N = self.suff_stats[cluster_i]

        return betaln(alpha + K, beta + N - K) - betaln(alpha, beta)

    def posterior_pred(self, x, cluster_i):
        """P(X=x|cluster with N items, K successes)"""
        alpha, beta = self.get_reparam_hypers()

        if cluster_i == len(self.suff_stats):
            K, N = 0, 0
        else:
            K, N = self.suff_stats[cluster_i]

        if x == 1:
            return np.log(alpha + K) - np.log(alpha + beta + N)
        else:
            return np.log(beta + N - K) - np.log(alpha + beta + N)

    def get_reparam_hypers(self):
        """reparameterize strength and balance into alpha and beta"""
        alpha = self.hypers[self.s_i] * self.hypers[self.b_i]
        beta = self.hypers[self.s_i] * (1.0 - self.hypers[self.b_i])

        return alpha, beta

    def posterior_param_sample(self, cluster_i, n_samples):
        alpha, beta = self.get_reparam_hypers()
        K, N = self.suff_stats[cluster_i]

        return dists.beta(alpha + K, beta + N - K).rvs(n_samples)

    def _transition_hypers(self):
        """Deprecated transition method. Not completely wrong... but there are more correct ways of doing this"""
        def beta_sufficient_stats():
            params_arr = np.array(
                [
                    self.posterior_pred(x=1, cluster_i=cluster_i)
                    for cluster_i in range(len(self.suff_stats))
                ]
            )
            params_arr = np.exp(params_arr)

            # TODO is this faster or slower than just iterating?
            params_arr = np.where(np.isclose(params_arr, 1), 0.999, params_arr)
            params_arr = np.where(np.isclose(params_arr, 0), 0.001, params_arr)

            sum_log_x = np.sum(np.log(params_arr))
            sum_minus_log_x = np.sum(np.log(1 - params_arr))

            assert sum_log_x != np.nan and sum_minus_log_x != np.nan

            return (sum_log_x, sum_minus_log_x, len(params_arr))

        def beta_log_lik(pop, *args):
            sum_log_x = args[0]
            sum_minus_log_x = args[1]
            K = args[2]

            alpha = pop["s"] * pop["b"]
            beta = pop["s"] * (1.0 - pop["b"])

            log_lik = 0
            log_lik -= K * betaln(alpha, beta)
            log_lik += (beta - 1.0) * sum_minus_log_x
            log_lik += (alpha - 1.0) * sum_log_x

            return log_lik

        hypers = [
            Param(name="s", prior=self.s_dist),
            Param(name="b", prior=self.b_dist),
        ]

        mod = Model(
            name="fun",
            params=hypers,
            like_fun=beta_log_lik,
            like_args=beta_sufficient_stats(),
            initial_zeros_ok=False,
            use_priors=True,
            verbose=False,
        )

        burnin = 200
        mod(300, burnin=False)
        self.hypers = utils.posterior_sample(mod, burnin)