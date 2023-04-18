from .dim import Dim
import numpy as np
from scipy.special import gammaln
from RunDEMC import Model, Param, dists
import utils

LOG2 = np.log(2)
LOGPI = np.log(np.pi)
LOG2PI = LOG2 + LOGPI

"""DEPRECATED"""
class Normal(Dim):
    """Dimension object for continuous data.

    data ~ Normal(mu, 1/rho)

    Prior:
    mu ~ Normal(m, 1/r*rho)
    rho ~ Gamma(nu/2, s/2)

    Hyperprior:
    [r, nu, m, s]

    Sufficient Statistics:
    [sum_x, sum_x_sq, N]

    https://github.com/probcomp/cgpm/blob/master/src/primitives/normal.py
    """
    r_i = 0  # relative precision of mu
    nu_i = 1  # df of precision of rho
    m_i = 2  # mean of mu
    s_i = 3  # mean of rho is nu/s

    sum_x_i = 0
    sum_x_sq_i = 1
    N_i = 2

    def __init__(self, index, data, suff_stats, hypers=None, rng=None):
        # distribution of hyperparams (fixed)
        mean = np.mean(data)
        std = np.std(data)

        # from baxcat_cxx
        self.r_dist = dists.gamma(1, std)
        self.nu_dist = dists.gamma(2, .5)
        self.m_dist = dists.normal(mean, std)
        self.s_dist = dists.gamma(1, std)

        super().__init__(index, data, suff_stats, hypers, rng)

    def add_to_cluster(self, x, cluster_i):
        """add an item to a cluster, return True if added to a new cluster. False otherwise"""
        assert cluster_i <= len(self.suff_stats)

        if cluster_i < len(self.suff_stats):
            self.suff_stats[cluster_i][self.sum_x_i] += x
            self.suff_stats[cluster_i][self.sum_x_sq_i] += x**2
            self.suff_stats[cluster_i][self.N_i] += 1
            
            return False

        elif cluster_i == len(self.suff_stats):
            self.suff_stats.append([x, x**2, 1])
            
            return True

    def remove_from_cluster(self, x, cluster_i, keep=False):
        assert self.suff_stats[cluster_i][self.N_i] != 0

        self.suff_stats[cluster_i][self.sum_x_i] -= x
        self.suff_stats[cluster_i][self.sum_x_sq_i] -= x**2
        self.suff_stats[cluster_i][self.N_i] -= 1

        # only allowed to delete the cluster if given permission
        if not keep and self.suff_stats[cluster_i][self.N_i] == 0:
            del self.suff_stats[cluster_i]

    def gen_hypers(self):
        r = self.r_dist.rvs(1, self.rng)[0] # relative precision of mu
        nu = self.nu_dist.rvs(1, self.rng)[0] # df of precision of rho
        m = self.m_dist.rvs(1, self.rng)[0] # mean of mu
        s = self.s_dist.rvs(1, self.rng)[0] # mean of rho is nu/s

        return [r, nu, m, s]

    def p_hypers(self):
        p_r = self.r_dist.logpdf(x=self.hypers[self.r_i]) 
        p_nu = self.nu_dist.logpdf(x=self.hypers[self.nu_i]) 
        p_m = self.m_dist.logpdf(x=self.hypers[self.m_i]) 
        p_s = self.s_dist.logpdf(x=self.hypers[self.s_i]) 
        
        return p_r + p_nu + p_m + p_s

    def transition_hypers(self):
        def log_lik(pop, *args):
            liks = 0
            hypers = [pop["r"], pop["nu"], pop["m"], pop["s"]]

            for cluster_i in range(len(self.suff_stats)):
                liks += self.marginal_lik(cluster_i, hypers)

            return liks

        hypers = [
            Param(name="r", prior=self.r_dist),
            Param(name="nu", prior=self.nu_dist),
            Param(name="m", prior=self.m_dist),
            Param(name="s", prior=self.s_dist),
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
        mod(400, burnin=False)
        utils.plot_loglik(mod)
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
        if hypers is None:
            hypers = self.hypers
        r, nu, _m, s = hypers
        
        N = self.suff_stats[cluster_i][self.N_i]

        rn, nun, _mn, sn  = self.get_posterior_hypers(cluster_i, hypers)

        Z0 = Normal.calc_log_Z(r, nu, s)
        ZN = Normal.calc_log_Z(rn, nun, sn)

        prob = -(N / 2.0) * LOG2PI + ZN - Z0

        return prob

    def posterior_pred(self, x, cluster_i):
        rn, nun, _mn, sn = self.get_posterior_hypers(cluster_i)

        created_new = self.add_to_cluster(x, cluster_i) # TODO: this is very jank
        rm, num, _mm, sm = self.get_posterior_hypers(cluster_i)
        self.remove_from_cluster(x, cluster_i, keep=(not created_new))
        
        ZN = Normal.calc_log_Z(rn, nun, sn)
        ZM = Normal.calc_log_Z(rm, num, sm)
        
        return -0.5 * LOG2PI + ZM - ZN

    def get_posterior_hypers(self, cluster_i, hypers=None):
        if hypers is None:
            hypers = self.hypers
        r, nu, m, s = hypers

        if cluster_i == len(self.suff_stats):
            sum_x = 0
            sum_x_sq = 0
            N = 0
        else:
            sum_x, sum_x_sq, N = self.suff_stats[cluster_i]

        rn = r + float(N)
        nun = nu + float(N)
        mn = (r * m + sum_x) / rn
        sn = s + sum_x_sq + r * m * m - rn * mn * mn
        
        if type(sn) == np.ndarray:
            sn[sn == 0] = s[sn == 0]
        else:
            if sn == 0:
                sn = s

        return rn, nun, mn, sn

    @staticmethod
    def calc_log_Z(r, nu, s):
        """normalization constant"""
        assert np.all(r) > 0
        assert np.all(nu) > 0
        assert np.all(s) > 0

        return (
            ((nu + 1) / 2) * np.log(2)
            + 0.5 * np.log(np.pi)
            - 0.5 * np.log(r)
            - (nu / 2) * np.log(s)
            + gammaln(nu / 2)
        )
