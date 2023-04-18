from abc import ABC, abstractmethod
from numpy.random import default_rng


class Dim(ABC):
    """a Dim object, associated with a column of the dataframe"""

    def __init__(self, index, data, suff_stats, hypers=None, rng=None):
        """
        Args:
            index (int): index of the Dim in the dataframe. mostly exists so that Dims can be hashed
            data (np.array.view()): the data associated with that Dim
            suff_stats (list(list)): each list contains the sufficient statistics for each row cluster
            hypers (list): list of hyperparamaters
        """
        self.rng = rng
        if self.rng is None:
            self.rng = default_rng()

        self.hypers = hypers
        if hypers is None:
            self.hypers = self.gen_hypers()

        self.index = index
        self.data = data
        self.suff_stats = suff_stats

    @abstractmethod
    def add_to_cluster(self, x, cluster_i):
        """update sufficient statistics for x added to cluster_i"""
        pass

    @abstractmethod
    def remove_from_cluster(self, x, cluster_i, keep=False):
        """update sufficient statistics for x removed from cluster_i"""
        pass

    @abstractmethod
    def gen_hypers(self):
        pass

    @abstractmethod
    def p_hypers(self):
        pass

    @abstractmethod
    def transition_hypers(self):
        pass

    @abstractmethod
    def set_suff_stats(self, clusters):
        """compute and set all sufficient statistics for the clustering scheme"""
        pass

    @abstractmethod
    def marginal_lik(self):
        pass

    @abstractmethod
    def posterior_pred(self):
        pass

    @abstractmethod
    def posterior_param_sample(self, cluster_i, n_samples):
        """sample from the posterior parameter distribution"""
        pass

    def __hash__(self):
        return hash(self.index)

    def __repr__(self):
        s = f"""Dim {self.index} ({type(self).__name__})\nHypers: {self.hypers}"""
        
        return s
