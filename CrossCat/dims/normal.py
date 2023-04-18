from .dim import Dim
import numpy as np

from scipy.special import gammaln

from RunDEMC import Model, Param, dists
import utils

LOG2 = np.log(2)
LOGPI = np.log(np.pi)
LOG2PI = LOG2 + LOGPI


class Normal(Dim):
    """Dimension object for continuous data.

    data ~ Normal(mu, sigma^2)

    Prior:
    mu | tau ~ Normal(m, sigma^2/k)
    sigma^2 ~ invGamma(a, b)

    Hyperprior:
    m ~ Normal(mean data, var. data)
    k ~ Gamma(1, std) -->  k is the effective sample size

    a ~ Gamma(2, 1)
    b ~ Gamma(1, std)

    [m, k, a, b]

    Sufficient Statistics:
    [sum_x, sum_x_sq, N]

    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    """

    m_i = 0  # mean of the prior mean
    k_i = 1  # std of the prior mean
    a_i = 2  # mean of prior variance
    b_i = 3  # std of prior variance

    sum_x_i = 0
    sum_x_sq_i = 1
    N_i = 2

    def __init__(self, index, data, suff_stats, hypers=None, rng=None):
        # distribution of hyperparams (fixed)
        mean = np.mean(data)
        std = np.std(data)

        self.m_dist = dists.normal(mean, std)
        self.k_dist = dists.gamma(1, std)
        self.a_dist = dists.gamma(2, 1)
        self.b_dist = dists.gamma(1, std)

        super().__init__(index, data, suff_stats, hypers, rng)

    def add_to_cluster(self, x, cluster_i):
        assert cluster_i <= len(self.suff_stats)

        if cluster_i < len(self.suff_stats):
            self.suff_stats[cluster_i][self.sum_x_i] += x
            self.suff_stats[cluster_i][self.sum_x_sq_i] += x**2
            self.suff_stats[cluster_i][self.N_i] += 1

        elif cluster_i == len(self.suff_stats):
            self.suff_stats.append([x, x**2, 1])

    def remove_from_cluster(self, x, cluster_i, keep=False):
        assert self.suff_stats[cluster_i][self.N_i] != 0

        self.suff_stats[cluster_i][self.sum_x_i] -= x
        self.suff_stats[cluster_i][self.sum_x_sq_i] -= x**2
        self.suff_stats[cluster_i][self.N_i] -= 1

        # only allowed to delete the cluster if given permission
        if not keep and self.suff_stats[cluster_i][self.N_i] == 0:
            del self.suff_stats[cluster_i]

    def gen_hypers(self):
        m = self.m_dist.rvs(1, self.rng)[0]
        k = self.k_dist.rvs(1, self.rng)[0]
        a = self.a_dist.rvs(1, self.rng)[0]
        b = self.b_dist.rvs(1, self.rng)[0]

        return m, k, a, b

    def p_hypers(self):
        p_m = self.m_dist.logpdf(x=self.hypers[self.m_i])
        p_k = self.k_dist.logpdf(x=self.hypers[self.k_i])
        p_a = self.a_dist.logpdf(x=self.hypers[self.a_i])
        p_b = self.b_dist.logpdf(x=self.hypers[self.b_i])

        return p_m + p_k + p_a + p_b

    def transition_hypers(self, MAP=False, n_samps=100):
        def log_lik(pop, *args):
            liks = 0
            hypers = [pop["m"], pop["k"], pop["a"], pop["b"]]

            for cluster_i in range(len(self.suff_stats)):
                liks += self.marginal_lik(cluster_i, hypers)

            return liks

        hypers = [
            Param(name="m", prior=self.m_dist),
            Param(name="k", prior=self.k_dist),
            Param(name="a", prior=self.a_dist),
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
        mod(burnin + n_samps, burnin=False)
        # utils.plot_loglik(mod)

        if MAP:
            self.hypers = utils.MAP_estimates(mod, burnin)
        else:
            self.hypers = utils.posterior_sample(mod, burnin)

    def set_suff_stats(self, clusters):
        self.suff_stats = [
            [
                np.sum(self.data[list(cluster)]),
                np.sum(self.data[list(cluster)] ** 2),
                len(cluster),
            ]
            for cluster in clusters
        ]
        # [sum_x, sum_x_sq, N]

    def marginal_lik(self, cluster_i, hypers=None):
        """Marginal likelihood for this cluster with optional hyperparams. If None, defaults to current hyperparams."""
        if hypers is None:
            hypers = self.hypers

        N = self.suff_stats[cluster_i][self.N_i]
        hypers_n = Normal.get_posterior_hypers(self.suff_stats[cluster_i], hypers)

        Z0 = Normal.calc_log_Z(hypers)
        ZN = Normal.calc_log_Z(hypers_n)

        prob = -(N / 2.0) * LOG2PI + ZN - Z0

        return prob

    def posterior_pred(self, x, cluster_i):
        if cluster_i == len(self.suff_stats):
            suff_stats = [0, 0, 0]
        else:
            suff_stats = self.suff_stats[cluster_i]

        m_n, k_n, a_n, b_n = Normal.get_posterior_hypers(suff_stats, self.hypers)

        std = b_n * (k_n + 1) / (a_n * k_n)

        return dists.students_t(mean=m_n, std=std, df=2 * a_n).logpdf(x)

    def posterior_param_sample(self, cluster_i, n_samples):
        m, k, a, b = Normal.get_posterior_hypers(self.suff_stats[cluster_i], self.hypers)

        vars = dists.invgamma(a, b).rvs(n_samples)
        means = dists.normal(m, vars/k).rvs() 

        return means, np.sqrt(vars)

    @staticmethod
    def calc_log_Z(hypers):
        _m, k, a, b = hypers

        return gammaln(a) - (a * np.log(b)) + (LOG2PI - np.log(k)) / 2

    @staticmethod
    def get_posterior_hypers(suff_stats, hypers):
        sum_x, sum_x_sq, N = suff_stats

        if N == 0:
            return hypers

        var = (sum_x_sq / N) - ((sum_x / N) ** 2)

        m_0, k_0, a_0, b_0 = hypers

        m_n = (k_0 * m_0 + sum_x) / (k_0 + N)
        k_n = k_0 + N
        a_n = a_0 + N / 2

        num = (k_0 * N) * ((sum_x / N - m_0) ** 2)
        den = 2 * (k_0 + N)
        b_n = b_0 + (N * var) / 2 + num / den

        return m_n, k_n, a_n, b_n

    def _marginal_lik(self, cluster_i, hypers=None):
        """equivalent to used one - for testing"""
        if hypers is None:
            hypers = self.hypers

        N = self.suff_stats[cluster_i][self.N_i]

        _m_0, k_0, a_0, b_0 = hypers
        _m_n, k_n, a_n, b_n = Normal.get_posterior_hypers(
            self.suff_stats[cluster_i], hypers
        )

        return (
            gammaln(a_n)
            - gammaln(a_0)
            + a_0 * np.log(b_0)
            - a_n * np.log(b_n)
            + ((np.log(k_0) - np.log(k_n)) / 2)
            - (N / 2) * LOG2PI
        )
