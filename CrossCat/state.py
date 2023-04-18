import numpy as np

import matplotlib.pyplot as plt

import utils
from copy import deepcopy

from crp import CRP
from view import View

from dims.bernoulli import Bernoulli
from dims.normal import Normal


class State(CRP):
    dim_dists = {"bern": Bernoulli, "normal": Normal}

    def __init__(self, data, dists, alpha=None, views=None, cats=None, rng=None):
        """Initialize a state with specific params, sampling missing ones from the priors

        Args:
            data (array): Data matrix
            dists (list): specified probability distribution for each dimension
            alpha (float, optional): outer CRP concentration parameter. Defaults to None.
            views (list(set), optional): a list of sets, each containing the column indeces in a view. Defaults to None.
            cats (list(list(set)), optional): specifies the categorization for each row in each view using above format ^. Defaults to None.
            rng (Generator, optional): numpy rng object. Defaults to None.
        """
        super().__init__(alpha, rng)
        #self.alpha_dist = RunDEMC.dists.gamma(2, 1)

        self.n_obs, self.n_dims = data.shape
        self.data = data

        # create dims, dim hypers
        self.dims = []
        for dim_i, dist in enumerate(dists):
            dim = self.dim_dists[dist](
                index=dim_i, data=data[:, dim_i].view(), suff_stats=None
            )
            self.dims.append(dim)

        # assign dims to views
        if views is None:
            self.assign(self.dims)
        else:
            self.elements = [{self.dims[dim_i] for dim_i in view} for view in views]
            self.K = len(views)
            self.N = self.n_dims

        # create views objects, categories + associated dim parameters
        self.views = []
        for dims in self.elements:
            view = View(dims=dims)

            if cats is None:
                view.assign(range(self.n_obs))
            else:
                view.elements = cats[len(self.views)]
                view.K = len(cats[len(self.views)])
                view.N = self.n_obs

            for dim in view.dims:
                dim.set_suff_stats(view.elements)

            self.views.append(view)

    def transition(self):
        self.transition_alphas()
        self.transition_hypers()
        self.transition_clusters()
        self.transition_dims()
        # self.transition_params()

    def transition_alphas(self, MAP=False, n_samps=100):
        """new MCMC method"""
        self.transition_alpha(MAP, n_samps)

        for view in self.views:
            view.transition_alpha()

    def transition_hypers(self, MAP=False, n_samps=100):
        """transition cluster-parameter generating hyperparameters for each dimension"""
        for dim in self.dims:
            dim.transition_hypers(MAP, n_samps)

    def transition_clusters(self):
        for view in self.views:
            view.transition_clusters()

    def transition_dims(self):
        """original transition dims function"""
        random_order_dims = self.rng.permutation(self.n_dims)

        for dim_i in random_order_dims:
            dim = self.dims[dim_i]

            view_i = self.cluster_of(dim)
            # self._verify_views() # for testing

            # calc priors
            priors = self.calc_transition_log_probs(view_i)

            # calc likes:
            log_liks = [view.marginal_lik_dim(dim) for view in self.views]
            new_view = View(dims=None)
            new_view.assign(range(self.n_obs))

            log_liks.append(new_view.marginal_lik_dim(dim))

            probs = utils.exp_normalize(priors + np.array(log_liks))
            choice = self.rng.choice(len(probs), p=probs)

            # dims didn't change spots
            if choice == view_i:
                continue

            # dim belongs to a brand new category
            if choice == self.K:
                self.add_obj(dim, choice)

                dim.set_suff_stats(new_view.elements)
                # I want a reference to the existing, not a new set
                new_view.dims = self.elements[choice]

                self.views.append(new_view)

                self.remove_obj(dim, view_i)

            # dim belongs to an existing category
            else:
                self.add_obj(dim, choice)
                dim.set_suff_stats(self.views[choice].elements)
                self.remove_obj(dim, view_i)

            # remove the old view from the views list if I deleted a view
            if len(self.views[view_i].dims) == 0:
                del self.views[view_i]

    def log_lik(self):
        """(original) log likelihood of the model"""
        log_lik = np.longdouble(0)

        # prior prob of observing each view
        log_lik += self.alpha_dist.logpdf(self.alpha)
        log_lik += self.calc_logpdf_marginal(self.alpha)

        for view in self.views:
            # prior prob of observing each cluster
            log_lik += self.alpha_dist.logpdf(view.alpha)
            log_lik += view.calc_logpdf_marginal(view.alpha)

            # joint posterior of the data in each cluster
            for dim in view.dims:
                log_lik += dim.p_hypers()
                log_lik += view.marginal_lik_dim(dim)

        return log_lik

    def plot(self):
        fig, axs = plt.subplots(1, len(self.views))
        vmin = np.amin(self.data)
        vmax = np.amax(self.data)

        for i, view in enumerate(self.views):
            if len(self.views) == 1:
                ax = plt.subplot()
            else:
                ax = axs[i]

            dim_indeces = view.get_dim_indeces()

            subset = self.data[:, dim_indeces]
            to_graph = np.zeros_like(subset)

            # construct matrix for the view
            cur = 0
            row_order = []
            dividing_lines = []
            for cluster in view.elements:
                to_graph[range(cur, cur + len(cluster)), :] = subset[list(cluster)]
                cur += len(cluster)
                row_order += list(cluster)
                dividing_lines.append(cur - 0.5)

            for line in dividing_lines:
                ax.axhline(y=line, color="r", linestyle="-")

            ax.xaxis.set_ticks(range(len(dim_indeces)))
            ax.set_xticklabels(dim_indeces)
            ax.yaxis.set_ticks(range(self.n_obs))
            ax.set_yticklabels(row_order)

            ax.matshow(to_graph, vmin=vmin, vmax=vmax)
        
        fig.tight_layout()

    def _verify_views(self):
        for view, dims in zip(self.views, self.elements):
            assert view.dims == dims, f"objects {view.dims}, {dims} should be the same"

        assert len(self.views) == len(
            self.elements
        ), f"views crp {len(self.elements)} and list {len(self.views)} should be the same length"