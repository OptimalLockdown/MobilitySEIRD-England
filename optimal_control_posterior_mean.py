import copy
import logging
import time

import numpy as np
import pylab as plt
from abcpy.backends import BackendDummy, BackendMPI
from abcpy.continuousmodels import Uniform
from abcpy.output import Journal
from scipy import optimize

from src.models import SEI4RD
from src.optimal_control_utils import obtain_alphas_from_mobility, different_bounds
from src.utils import weighted_quantile, plot_confidence_bands


class PosteriorCost:

    def __init__(self, model, phi_func_sc, phi_func_death, beta_vals, kappa_vals, gamma_c_vals, gamma_r_vals,
                 gamma_rc_vals, nu_vals,
                 rho_vals, rho_prime_vals, alpha_123_vals, alpha_4_vals, alpha_5_vals, initial_exposed_vals,
                 weights=None, loss="Isc"):
        """ This class stores a set of posterior parameter samples (with optional associated weights); it also stores
        the state of the epidemics corresponding to the different parameters, and implements methods for computing the
        expected cost, as well as updating the state of the model.

         All the parameters are arrays with shape [n_samples, param_size].
        This also takes care of simulating the model and setting the states to the final state at the end of the
        training period.
        :weight is the weight of the samples which are used in the computation of the expected cost over posterior.
        """
        self.model = model
        self.phi_func_sc = phi_func_sc
        self.phi_func_death = phi_func_death
        self.beta_vals = beta_vals
        self.kappa_vals = kappa_vals
        self.gamma_c_vals = gamma_c_vals
        self.gamma_r_vals = gamma_r_vals
        self.gamma_rc_vals = gamma_rc_vals
        self.nu_vals = nu_vals
        self.rho_vals = rho_vals
        self.rho_prime_vals = rho_prime_vals
        self.alpha_123_vals = alpha_123_vals
        self.alpha_4_vals = alpha_4_vals
        self.alpha_5_vals = alpha_5_vals
        self.initial_exposed_vals = initial_exposed_vals
        if weights is None:
            self.weights = np.ones(len(beta_vals))
        else:
            self.weights = weights
        self.loss = loss
        if self.loss not in ("Isc", "deaths_Isc"):
            raise NotImplementedError

        # now we need to save the state variables for each of the posterior samples; use lists here as they are needed
        # to parallelize
        self.susceptible_states = np.zeros((self.weights.shape[0], 5))
        self.exposed_states = np.zeros((self.weights.shape[0], 5))
        self.infected_c1_states = np.zeros((self.weights.shape[0], 5))
        self.infected_c2_states = np.zeros((self.weights.shape[0], 5))
        self.infected_sc1_states = np.zeros((self.weights.shape[0], 5))
        self.infected_sc2_states = np.zeros((self.weights.shape[0], 5))
        self.removed_states = np.zeros((self.weights.shape[0], 5))
        self.deceased_states = np.zeros((self.weights.shape[0], 5))
        self.confirmed_states = np.zeros((self.weights.shape[0], 5))

        print("Evolve epidemics for the training period")
        for i in range(self.weights.shape[0]):
            pars = [self.beta_vals[i], 1.0 / self.kappa_vals[i], 1.0 / self.gamma_c_vals[i], 1.0 / self.gamma_r_vals[i],
                    1.0 / self.gamma_rc_vals[i], 1.0 / self.nu_vals[i], *self.rho_vals[i], *self.rho_prime_vals[i],
                    self.initial_exposed_vals[i], self.alpha_123_vals[i], self.alpha_4_vals[i], self.alpha_5_vals[i]]

            observation = self.model.forward_simulate(pars, 1, return_clinical_categories=True)[0]

            # get the final conditions; the content of observation is:
            # susceptible_evolution, exposed_evolution, infected_c_evolution, infected_sc1_evolution, \
            # infected_sc2_evolution, removed_evolution, deceased_evolution, confirmed_evolution, \
            # infected_c1_evolution, infected_c2_evolution
            self.susceptible_states[i] = observation[0, -1]
            self.exposed_states[i] = observation[1, -1]
            self.infected_c1_states[i] = observation[-2, -1]
            self.infected_c2_states[i] = observation[-1, -1]
            self.infected_sc1_states[i] = observation[3, -1]
            self.infected_sc2_states[i] = observation[4, -1]
            self.removed_states[i] = observation[5, -1]
            self.deceased_states[i] = observation[6, -1]
            self.confirmed_states[i] = observation[7, -1]

        print("State at the end of the training period has been saved")

    def _compute_single_cost(self, data):
        beta, kappa, gamma_c, gamma_r, gamma_rc, nu, rho, rho_prime, alpha_123, alpha_4, alpha_5, \
        susceptible_state, exposed_state, infected_c1_state, infected_c2_state, infected_sc1_state, \
        infected_sc2_state, removed_state, deceased_state, confirmed_state = data
        # the broadcasted model is local to every worker, right?? So I should be able to set its state without
        # interfering with the others
        self.model.set_state(
            susceptible_state=susceptible_state, exposed_state=exposed_state,
            infected_c1_state=infected_c1_state, infected_c2_state=infected_c2_state,
            infected_sc1_state=infected_sc1_state, infected_sc2_state=infected_sc2_state,
            removed_state=removed_state, deceased_state=deceased_state,
            confirmed_state=confirmed_state)

        # need to modify the alphas taking into account the multiplier for alpha_123, and the values alpha_4,
        # alpha_5 for the two older age groups.
        alpha_school, alpha_work, alpha_other = \
            obtain_alphas_from_mobility(alpha_123, alpha_4, alpha_5, mobility_school=self.mobility_school_bds.value(),
                                        mobility_work=self.mobility_work_bds.value(),
                                        mobility_other=self.mobility_other_bds.value())
        if self.loss == "Isc":
            susceptible_evolution, exposed_evolution, infected_sc1_evolution, infected_sc2_evolution, = self.model.evolve_epidemics_partial_sc(
                self.n_days_bds.value() * 10, beta=beta, kappa=kappa,
                gamma_c=gamma_c, gamma_r=gamma_r, rho=rho,
                alpha_school=alpha_school,
                alpha_work=alpha_work, alpha_other=alpha_other,
                alpha_home=self.alpha_home_bds.value())
            infected_sc1_evolution = infected_sc1_evolution[:-1, :].reshape(self.n_days_bds.value(), 10, 5).mean(axis=1)
            cost_infections = self.model.compute_J_Ic(infected_c_evolution=infected_sc1_evolution[1:],
                                                      phi_func=self.phi_func_sc)
        elif self.loss == "deaths_Isc":

            # evolve epidemics
            susceptible_evolution, exposed_evolution, infected_c_evolution, infected_sc1_evolution, infected_sc2_evolution, \
            removed_evolution, deceased_evolution, confirmed_evolution, infected_c1_evolution, infected_c2_evolution = \
                self.model.evolve_epidemics(self.n_days_bds.value() * 10, beta=beta, kappa=kappa,
                                            gamma_c=gamma_c, gamma_r=gamma_r, gamma_rc=gamma_rc, nu=nu, rho=rho,
                                            rho_prime=rho_prime, alpha_school=alpha_school,
                                            alpha_work=alpha_work, alpha_home=self.alpha_home_bds.value(),
                                            alpha_other=alpha_other, return_clinical_categories=True)

            infected_sc1_evolution = infected_sc1_evolution[:-1, :].reshape(self.n_days_bds.value(), 10, 5).mean(axis=1)
            cost_infections_sc = self.model.compute_J_Ic(infected_c_evolution=infected_sc1_evolution[1:],
                                                         phi_func=self.phi_func_sc)
            deceased_evolution = deceased_evolution[:-1, :].reshape(self.n_days_bds.value(), 10, 5).mean(axis=1)
            cost_infections_death = self.model.compute_J_Ic(infected_c_evolution=np.diff(deceased_evolution, axis=0),
                                                            phi_func=self.phi_func_death)
            cost_infections = cost_infections_sc + cost_infections_death
        # susceptible_evolution, exposed_evolution, infected_sc1_evolution, infected_sc2_evolution, \
        # infected_c1_evolution, infected_c2_evolution, = self.model.evolve_epidemics_partial(
        #     self.n_days_bds.value() * 10, beta=beta, kappa=kappa,
        #     gamma_c=gamma_c, gamma_r=gamma_r, gamma_rc=gamma_rc, nu=nu, rho=rho,
        #     rho_prime=rho_prime, alpha_school=alpha_school,
        #     alpha_work=alpha_work, alpha_other=alpha_other,
        #     alpha_home=self.alpha_home_bds.value())

        # take the mean over the time steps in each day
        # infected_c1_evolution = infected_c1_evolution[:-1, :].reshape(self.n_days_bds.value(), 10, 5).mean(axis=1)
        # infected_c2_evolution = infected_c2_evolution[:-1, :].reshape(self.n_days_bds.value(), 10, 5).mean(axis=1)
        # cost_infections = self.model_bds.value().compute_J_Isc1(infected_sc1_evolution=infected_sc1_evolution[1:],
        #                                              phi_func=self.phi_func)
        # in order to compute the cost of R_T: use the contact matrix which has been computed with the above call of
        # evolve epidemics.
        if self.sigma_bds.value() > 0:
            cost_R_T = self.model.compute_R(beta, kappa, rho, rho_prime, gamma_c, gamma_r, gamma_rc, nu,
                                            self.model.contact_matrix_evolution[-1]) * self.sigma_bds.value()
        else:
            cost_R_T = 0
        # now compute J; we need to discard the first element as it is the initial condition, on which the control has no effect.
        return cost_infections + cost_R_T

    def compute_cost(self, mobilities, n_days, sigma, epsilon, backend):
        mobility_matrix = mobilities.reshape(3, n_days)  # 3 instead of 4 as we are not using alpha_home
        mobility_school = np.repeat(mobility_matrix[0, :].reshape(-1, ), 10)
        mobility_work = np.repeat(mobility_matrix[1, :].reshape(-1, ), 10)
        alpha_home = np.ones_like(mobility_school)
        mobility_other = np.repeat(mobility_matrix[2, :].reshape(-1, ), 10)

        # broadcast the objects that do not need to be split:
        self.mobility_school_bds = backend.broadcast(mobility_school)
        self.mobility_work_bds = backend.broadcast(mobility_work)
        self.mobility_other_bds = backend.broadcast(mobility_other)
        self.alpha_home_bds = backend.broadcast(alpha_home)
        self.n_days_bds = backend.broadcast(n_days)
        self.sigma_bds = backend.broadcast(sigma)

        # parallelize  # do these need to be python lists?? not sure...
        data = zip(self.beta_vals, self.kappa_vals, self.gamma_c_vals, self.gamma_r_vals, self.gamma_rc_vals,
                   self.nu_vals, self.rho_vals, self.rho_prime_vals, self.alpha_123_vals,
                   self.alpha_4_vals, self.alpha_5_vals, self.susceptible_states, self.exposed_states,
                   self.infected_c1_states,
                   self.infected_c2_states, self.infected_sc1_states, self.infected_sc2_states, self.removed_states,
                   self.deceased_states, self.confirmed_states)

        data_pds = backend.parallelize(list(data))

        cost_infection_pds = backend.map(self._compute_single_cost, data_pds)
        cost_infection = backend.collect(cost_infection_pds)  # this also considers already the cost of R_T

        # now take weighted sum:
        total_J_infections = np.sum(cost_infection * self.weights) / np.sum(self.weights)

        total_J = total_J_infections + self.model.compute_J_mobility(mobility_matrix, epsilon)

        # print(total_J)
        return total_J

    def update_states(self, n_days, mobilities):
        """Evolves the epidemics for that number of days and updates the state at the end, for all the different
        parameter samples"""
        mobility_matrix = mobilities.reshape(3, n_days)  # 3 instead of 4 as we are not using alpha_home
        mobility_school = np.repeat(mobility_matrix[0, :].reshape(-1, ), 10)
        mobility_work = np.repeat(mobility_matrix[1, :].reshape(-1, ), 10)
        alpha_home = np.ones_like(mobility_school)
        mobility_other = np.repeat(mobility_matrix[2, :].reshape(-1, ), 10)

        for i in range(self.weights.shape[0]):
            # set the previous initial conditions
            self.model.set_state(susceptible_state=self.susceptible_states[i], exposed_state=self.exposed_states[i],
                                 infected_c1_state=self.infected_c1_states[i],
                                 infected_c2_state=self.infected_c2_states[i],
                                 infected_sc1_state=self.infected_sc1_states[i],
                                 infected_sc2_state=self.infected_sc2_states[i],
                                 removed_state=self.removed_states[i], deceased_state=self.deceased_states[i],
                                 confirmed_state=self.confirmed_states[i])

            alpha_school, alpha_work, alpha_other = \
                obtain_alphas_from_mobility(self.alpha_123_vals[i], self.alpha_4_vals[i], self.alpha_5_vals[i],
                                            mobility_school=mobility_school, mobility_work=mobility_work,
                                            mobility_other=mobility_other)
            # evolve epidemics
            susceptible_evolution, exposed_evolution, infected_c_evolution, infected_sc1_evolution, infected_sc2_evolution, \
            removed_evolution, deceased_evolution, confirmed_evolution, infected_c1_evolution, infected_c2_evolution = \
                self.model.evolve_epidemics(n_days * 10, beta=self.beta_vals[i], kappa=self.kappa_vals[i],
                                            gamma_c=self.gamma_c_vals[i], gamma_r=self.gamma_r_vals[i],
                                            gamma_rc=self.gamma_rc_vals[i], nu=self.nu_vals[i], rho=self.rho_vals[i],
                                            rho_prime=self.rho_prime_vals[i], alpha_school=alpha_school,
                                            alpha_work=alpha_work, alpha_home=alpha_home,
                                            alpha_other=alpha_other, return_clinical_categories=True)

            self.susceptible_states[i] = susceptible_evolution[-1]
            self.exposed_states[i] = exposed_evolution[-1]
            self.infected_c1_states[i] = infected_c1_evolution[-1]
            self.infected_c2_states[i] = infected_c2_evolution[-1]
            self.infected_sc1_states[i] = infected_sc1_evolution[-1]
            self.infected_sc2_states[i] = infected_sc2_evolution[-1]
            self.removed_states[i] = removed_evolution[-1]
            self.deceased_states[i] = deceased_evolution[-1]
            self.confirmed_states[i] = confirmed_evolution[-1]

    def evolve_and_compute_R(self, n_days, mobilities, update_states=False):
        """Evolves the epidemics for that number of days and computes the values of R. Optionally updates the state
        at the end, for all the different parameter samples"""
        mobility_matrix = mobilities.reshape(3, n_days)  # 3 instead of 4 as we are not using alpha_home
        mobility_school = np.repeat(mobility_matrix[0, :].reshape(-1, ), 10)
        mobility_work = np.repeat(mobility_matrix[1, :].reshape(-1, ), 10)
        alpha_home = np.ones_like(mobility_school)
        mobility_other = np.repeat(mobility_matrix[2, :].reshape(-1, ), 10)

        R_evolution = []
        for i in range(self.weights.shape[0]):
            # set the previous initial conditions
            self.model.set_state(susceptible_state=self.susceptible_states[i], exposed_state=self.exposed_states[i],
                                 infected_c1_state=self.infected_c1_states[i],
                                 infected_c2_state=self.infected_c2_states[i],
                                 infected_sc1_state=self.infected_sc1_states[i],
                                 infected_sc2_state=self.infected_sc2_states[i],
                                 removed_state=self.removed_states[i], deceased_state=self.deceased_states[i],
                                 confirmed_state=self.confirmed_states[i])

            alpha_school, alpha_work, alpha_other = \
                obtain_alphas_from_mobility(self.alpha_123_vals[i], self.alpha_4_vals[i], self.alpha_5_vals[i],
                                            mobility_school=mobility_school, mobility_work=mobility_work,
                                            mobility_other=mobility_other)
            # evolve epidemics
            susceptible_evolution, exposed_evolution, infected_c_evolution, infected_sc1_evolution, infected_sc2_evolution, \
            removed_evolution, deceased_evolution, confirmed_evolution, infected_c1_evolution, infected_c2_evolution = \
                self.model.evolve_epidemics(n_days * 10, beta=self.beta_vals[i], kappa=self.kappa_vals[i],
                                            gamma_c=self.gamma_c_vals[i], gamma_r=self.gamma_r_vals[i],
                                            gamma_rc=self.gamma_rc_vals[i], nu=self.nu_vals[i], rho=self.rho_vals[i],
                                            rho_prime=self.rho_prime_vals[i], alpha_school=alpha_school,
                                            alpha_work=alpha_work, alpha_home=alpha_home,
                                            alpha_other=alpha_other, return_clinical_categories=True)
            R_evolution.append(self.model.compute_R_evolution(beta=self.beta_vals[i], kappa=self.kappa_vals[i],
                                                              gamma_c=self.gamma_c_vals[i],
                                                              gamma_r=self.gamma_r_vals[i],
                                                              gamma_rc=self.gamma_rc_vals[i],
                                                              nu=self.nu_vals[i], rho=self.rho_vals[i],
                                                              rho_prime=self.rho_prime_vals[i]))
            if update_states:
                self.susceptible_states[i] = susceptible_evolution[-1]
            self.exposed_states[i] = exposed_evolution[-1]
            self.infected_c1_states[i] = infected_c1_evolution[-1]
            self.infected_c2_states[i] = infected_c2_evolution[-1]
            self.infected_sc1_states[i] = infected_sc1_evolution[-1]
            self.infected_sc2_states[i] = infected_sc2_evolution[-1]
            self.removed_states[i] = removed_evolution[-1]
            self.deceased_states[i] = deceased_evolution[-1]
            self.confirmed_states[i] = confirmed_evolution[-1]

        return np.array(R_evolution)

    def produce_plot(self, mobilities, n_days):
        """Evolves the epidemics for that number of days and update the state at the end, for all the different
        parameter samples"""
        mobility_matrix = mobilities.reshape(3, n_days)  # 3 instead of 4 as we are not using alpha_home
        mobility_school = np.repeat(mobility_matrix[0, :].reshape(-1, ), 10)
        mobility_work = np.repeat(mobility_matrix[1, :].reshape(-1, ), 10)
        alpha_home = np.ones_like(mobility_school)
        mobility_other = np.repeat(mobility_matrix[2, :].reshape(-1, ), 10)

        fig, ax1 = plt.subplots(1, figsize=(6, 4))
        color = 'tab:red'
        CI_size = 99
        if self.model.T == 41:
            ax1.set_xticks([20, 51, 81, 112])
            ax1.set_xticklabels(['01/05', '01/06', '01/07', '01/08'])
        if self.model.T == 56:
            ax1.set_xticks([5, 36, 66, 97])
            ax1.set_xticklabels(['01/05', '01/06', '01/07', '01/08'])
        elif self.model.T == 71:
            ax1.set_xticks([19, 49, 80, 111])
            ax1.set_xticklabels(['01/06', '01/07', '01/08', '01/09'])
            # ax1.set_xticks([0, 9, 19, 29, 39, 49])
            # ax1.set_xticks(np.arange(51), minor=True)
            # ax1.set_xticklabels(['11/05', '20/05', '30/05', '10/06', '20/06', '30/06'])
        elif self.model.T == 83:
            ax1.set_xticks([7, 37, 68, 99])
            ax1.set_xticklabels(['01/06', '01/07', '01/08', '01/09'])
            # ax1.set_xticks([0, 9, 19, 29, 39, 47])
            # ax1.set_xticks(np.arange(48), minor=True)
            # ax1.set_xticklabels(['23/05', '02/06', '12/06', '22/06', '02/07', '11/07'])
            # ax1.set_xticks([0, 9, 19, 29, 39, 49, 59])
            # ax1.set_xticks(np.arange(64), minor=True)
            # ax1.set_xticklabels(['23/05', '02/06', '12/06', '22/06', '02/07', '12/07', '22/07'])
        elif self.model.T == 183: # not sure
            ax1.set_xticks([0, 30, 61, 91])  # not sure about this either
            ax1.set_xticklabels(['01/09', '01/10', '01/11', '01/12'])

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Hospitalized', color=color)
        if self.model.T != 41:
            ax1.set_ylim([0, 12000])

        infected_c_list = []

        for i in range(self.weights.shape[0]):
            # set the previous initial conditions
            self.model.set_state(susceptible_state=self.susceptible_states[i], exposed_state=self.exposed_states[i],
                                 infected_c1_state=self.infected_c1_states[i],
                                 infected_c2_state=self.infected_c2_states[i],
                                 infected_sc1_state=self.infected_sc1_states[i],
                                 infected_sc2_state=self.infected_sc2_states[i],
                                 removed_state=self.removed_states[i], deceased_state=self.deceased_states[i],
                                 confirmed_state=self.confirmed_states[i])

            alpha_school, alpha_work, alpha_other = \
                obtain_alphas_from_mobility(self.alpha_123_vals[i], self.alpha_4_vals[i], self.alpha_5_vals[i],
                                            mobility_school=mobility_school, mobility_work=mobility_work,
                                            mobility_other=mobility_other)
            # evolve epidemics
            susceptible_evolution, exposed_evolution, infected_c_evolution, infected_sc1_evolution, infected_sc2_evolution, \
            removed_evolution, deceased_evolution, confirmed_evolution, infected_c1_evolution, infected_c2_evolution = \
                self.model.evolve_epidemics(n_days * 10, beta=self.beta_vals[i], kappa=self.kappa_vals[i],
                                            gamma_c=self.gamma_c_vals[i], gamma_r=self.gamma_r_vals[i],
                                            gamma_rc=self.gamma_rc_vals[i], nu=self.nu_vals[i], rho=self.rho_vals[i],
                                            rho_prime=self.rho_prime_vals[i], alpha_school=alpha_school,
                                            alpha_work=alpha_work, alpha_home=alpha_home,
                                            alpha_other=alpha_other, return_clinical_categories=True)
            infected_c_list.append(np.sum(infected_c_evolution[:-1, :].reshape(n_days, 10, 5).mean(axis=1), axis=1))

            # ax1.plot(np.sum(infected_sc1_evolution, axis=1) + np.sum(infected_sc2_evolution, axis=1), 'b', alpha=0.4)
            # ax1.plot(np.sum(infected_c_evolution[:-1, :].reshape(n_days, 10, 5).mean(axis=1), axis=1), color=color,
            #          alpha=0.4)
        # infected_c_list = np.array(infected_c_list)

        plot_confidence_bands(np.array(infected_c_list), fig=fig, ax=ax1, outer_band=0, inner_band=CI_size,
                              alpha_median=0.6, alpha_inner=.3, color_median=color)

        # legend_elements = [Line2D([0], [0], color='b', lw=2, label='Subclinical'), Line2D([0], [0], color='k', lw=2, label='Clinical')]
        # ax1.legend(handles=legend_elements)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Mobility', color=color)  # we already handled the x-label with ax1
        ax2.set_ylim([0, 1])
        ax2.plot(mobility_matrix[1, :], linestyle='solid', label=r'$work$', color=color)
        ax2.plot(mobility_matrix[0, :], linestyle='dashed', label=r'$school$', color=color)
        ax2.plot(mobility_matrix[2, :], linestyle='dashdot', label=r'$other$', color=color)
        ax2.legend()
        ax2.tick_params(axis='y', labelcolor=color)

        return fig, ax1


def main(epsilon, sigma, filename_prefix, perform_standard_optimal_control=False, perform_iterative_strategy=True,
         use_sample_with_higher_weight=False, use_posterior_median=False, n_post_samples=None, shift_each_iteration=1,
         n_shifts=10, window_size=30, only_plot=False, plot_file=None, plot_days=None, loss="deaths_Isc",
         results_folder=None, journal_file_name=None, training_window_length=None, use_mpi=False,
         restart_at_index=None):
    """epsilon is an array with size 3, with order school, work, other
    If use_sample_with_higher_weight is True: we do the procedure with that only, no posterior expectation
    use_posterior_median: do the optimal control with the marginal posterior median.
    n_post_samples: for the posterior expectation. Ignored if use_sample_with_higher_weight or use_posterior_median is True,
    shift_each_iteration and n_shifts are for the iterative strategy.
    """
    if use_mpi:
        print("Using MPI")
        backend = BackendMPI()
    else:
        backend = BackendDummy()

    print("Epsilon: ", epsilon)

    logging.basicConfig(level=logging.INFO)
    ############################ Load relevant data #################################################
    if results_folder is None:
        results_folder = "results/SEI4RD_england_infer_1Mar_31Aug/"
    data_folder = "data/england_inference_data_1Mar_to_31Aug/"

    alpha_home = 1  # set this to 1
    mobility_work = np.load(data_folder + "mobility_work.npy")
    mobility_other = np.load(data_folder + "mobility_other.npy")
    mobility_school = np.load(data_folder + "mobility_school.npy")

    england_pop = np.load(data_folder + "england_pop.npy", allow_pickle=True)

    contact_matrix_home_england = np.load(data_folder + "contact_matrix_home_england.npy")
    contact_matrix_work_england = np.load(data_folder + "contact_matrix_work_england.npy")
    contact_matrix_school_england = np.load(data_folder + "contact_matrix_school_england.npy")
    contact_matrix_other_england = np.load(data_folder + "contact_matrix_other_england.npy")

    if journal_file_name is None:
        jrnl = Journal.fromFile(results_folder + "PMCABC_inf3.jrl")
    else:
        jrnl = Journal.fromFile(results_folder + journal_file_name)
    #################################### Define Model #################################################
    # parameters
    n = 5  # number of age groups
    dt = 0.1  # integration timestep
    if training_window_length is not None:
        T = training_window_length
    else:
        T = mobility_school.shape[0] - 1  # horizon time in days
    total_population = england_pop  # population for each age group
    # 16th March: Boris Johnson asked old people to isolate; we then learn a new alpha from the 18th March:
    lockdown_day = 17

    # alpha_home = np.repeat(alpha_home, np.int(1 / dt), axis=0)
    mobility_work = np.repeat(mobility_work[0:T + 1], np.int(1 / dt), axis=0)
    mobility_other = np.repeat(mobility_other[0:T + 1], np.int(1 / dt), axis=0)
    mobility_school = np.repeat(mobility_school[0:T + 1], np.int(1 / dt), axis=0)
    # daily_tests = np.repeat(daily_tests, np.int(1 / dt), axis=0)

    # ABC model (priors need to be fixed better):
    beta = Uniform([[0], [0.5]], name='beta')  # controls how fast the epidemics grows. Related to R_0
    d_L = Uniform([[1], [16]], name='d_L')  # average duration of incubation
    d_C = Uniform([[1], [16]], name='d_C')  # average time before going to clinical
    d_R = Uniform([[1], [16]], name='d_R')  # average recovery time
    d_RC = Uniform([[1], [16]], name='d_RC')  # average recovery time
    d_D = Uniform([[1], [16]], name='d_D')  # average duration of infected clinical state (resulting in death)
    p01 = Uniform([[0], [1]], name="p01")
    p02 = Uniform([[0], [1]], name="p02")
    p03 = Uniform([[0], [1]], name="p03")
    p04 = Uniform([[0], [1]], name="p04")
    p05 = Uniform([[0], [1]], name="p05")
    p11 = Uniform([[0], [1]], name="p11")
    p12 = Uniform([[0], [1]], name="p12")
    p13 = Uniform([[0], [1]], name="p13")
    p14 = Uniform([[0], [1]], name="p14")
    p15 = Uniform([[0], [1]], name="p15")
    initial_exposed = Uniform([[0], [500]], name="initial_exposed")
    alpha_123 = Uniform([[0.3], [1]], name="alpha_123")
    alpha_4 = Uniform([[0], [1]], name="alpha_4")
    alpha_5 = Uniform([[0], [1]], name="alpha_5")

    model = SEI4RD([beta, d_L, d_C, d_R, d_RC, d_D, p01, p02, p03, p04, p05, p11, p12, p13, p14, p15, initial_exposed,
                    alpha_123, alpha_4, alpha_5], tot_population=total_population, T=T,
                   contact_matrix_school=contact_matrix_school_england, contact_matrix_work=contact_matrix_work_england,
                   contact_matrix_home=contact_matrix_home_england, contact_matrix_other=contact_matrix_other_england,
                   alpha_school=mobility_school, alpha_work=mobility_work, alpha_home=alpha_home,
                   alpha_other=mobility_other, modify_alpha_home=False, dt=dt, return_once_a_day=True,
                   learn_alphas_old=True, lockdown_day=lockdown_day)

    # guess for a phi function
    NHS_max = 10000

    def phi_func_sc(x):  # this is an hard max function.
        return np.maximum(0, x - NHS_max)

    def phi_func_death(x):  # this is an hard max function.
        return np.maximum(0, x)

    # def phi_func(x):
    #     return np.pow(np.maximum(0, x - NHS_max), 2)

    # def phi_func(x, beta=.1):  # this is the softplus, a smooth version of hard max
    #    threshold = 30
    #    shape = x.shape
    #    x = x.reshape(-1)
    #    new_x = x - NHS_max
    #    indices = new_x * beta < threshold
    #    phi_x = copy.deepcopy(new_x)  # is deepcopy actually needed?
    #    phi_x[indices] = np.log(
    #        1 + np.exp(new_x[indices] * beta)) / beta  # approximate for numerical stability in other places
    #    return phi_x.reshape(shape)

    # extract posterior sample points and bootstrap them:
    seed = 1
    np.random.seed(seed)
    iteration = - 1
    weights = jrnl.get_weights(iteration) / np.sum(jrnl.get_weights(iteration))
    params = jrnl.get_parameters(iteration)
    if not use_posterior_median:
        if use_sample_with_higher_weight:
            post_samples = np.where(weights == weights.max())[0]
        else:
            # bootstrap
            if n_post_samples is None:
                n_post_samples = len(weights)
            post_samples = np.random.choice(range(len(weights)), p=weights.reshape(-1), size=n_post_samples)

        beta_values = np.array([params['beta'][i][0] for i in post_samples])
        kappa_values = np.array([1 / params['d_L'][i][0] for i in post_samples])
        gamma_c_values = np.array([1 / params['d_C'][i][0] for i in post_samples])
        gamma_r_values = np.array([1 / params['d_R'][i][0] for i in post_samples])
        gamma_rc_values = np.array([1 / params['d_RC'][i][0] for i in post_samples])
        nu_values = np.array([1 / params['d_D'][i][0] for i in post_samples])
        rho_values = np.array(
            [np.array([params[key][i][0] for key in ['p01', 'p02', 'p03', 'p04', 'p05']]).reshape(-1) for i in
             post_samples])
        rho_prime_values = np.array(
            [np.array([params[key][i][0] for key in ['p11', 'p12', 'p13', 'p14', 'p15']]).reshape(-1) for i in
             post_samples])
        alpha_123_values = np.array([params["alpha_123"][i][0] for i in post_samples])
        alpha_4_values = np.array([params["alpha_4"][i][0] for i in post_samples])
        alpha_5_values = np.array([params["alpha_5"][i][0] for i in post_samples])
        initial_exposed_values = np.array([params["initial_exposed"][i][0] for i in post_samples])
    else:
        params_array = np.array([[params[key][i] for i in range(len(params[key]))] for key in params.keys()]).squeeze()
        marginal_medians = {key: weighted_quantile(np.array(params[key]).reshape(-1), [0.5], weights.squeeze()) for i in
                            range(params_array.shape[0]) for key in params.keys()}

        beta_values = np.array([marginal_medians['beta'][0]])
        kappa_values = np.array([1 / marginal_medians['d_L'][0]])
        gamma_c_values = np.array([1 / marginal_medians['d_C'][0]])
        gamma_r_values = np.array([1 / marginal_medians['d_R'][0]])
        gamma_rc_values = np.array([1 / marginal_medians['d_RC'][0]])
        nu_values = np.array([1 / marginal_medians['d_D'][0]])
        rho_values = np.array(
            [np.array([marginal_medians[key][0] for key in ['p01', 'p02', 'p03', 'p04', 'p05']]).reshape(-1)])
        rho_prime_values = np.array(
            [np.array([marginal_medians[key][0] for key in ['p11', 'p12', 'p13', 'p14', 'p15']]).reshape(-1)])
        alpha_123_values = np.array([marginal_medians["alpha_123"][0]])
        alpha_4_values = np.array([marginal_medians["alpha_4"][0]])
        alpha_5_values = np.array([marginal_medians["alpha_5"][0]])
        initial_exposed_values = np.array([marginal_medians["initial_exposed"][0]])

    # instantiate the posterior cost class:
    posterior_cost = PosteriorCost(model, phi_func_sc=phi_func_sc, phi_func_death=phi_func_death, beta_vals=beta_values,
                                   kappa_vals=kappa_values,
                                   gamma_c_vals=gamma_c_values, gamma_r_vals=gamma_r_values,
                                   gamma_rc_vals=gamma_rc_values, nu_vals=nu_values, rho_vals=rho_values,
                                   rho_prime_vals=rho_prime_values, alpha_123_vals=alpha_123_values,
                                   alpha_4_vals=alpha_4_values, alpha_5_vals=alpha_5_values,
                                   initial_exposed_vals=initial_exposed_values, loss=loss)

    if plot_days is None:
        n_days = 120
    else:
        n_days = plot_days
    end_training_mobility_values = [mobility_school[-1], mobility_work[-1], mobility_other[-1]]
    # alpha initial is taken assuming values will be kept constant as it was on the last day observed
    mobility_initial = copy.deepcopy(
        np.stack((mobility_school[-1] * np.ones(shape=(n_days,)), mobility_work[-1] * np.ones(shape=(n_days,)),
                  mobility_other[-1] * np.ones(shape=(n_days,))))).flatten()

    # Only plot using a mobility file
    if only_plot:
        mobility = np.load(results_folder + plot_file)[:, 0:n_days]
        fig, ax = posterior_cost.produce_plot(mobility, n_days)
        plt.savefig(results_folder + filename_prefix + ".pdf")
        plt.close(fig)
        return

    # try cost computation:
    t = time.time()
    cost_initial = posterior_cost.compute_cost(mobility_initial, n_days, sigma, epsilon, backend)
    # fig, ax = posterior_cost.produce_plot(mobility_initial, n_days)
    # plt.savefig(results_folder + filename_prefix + "evolution_under_final_training_lockdown_conditions.pdf")
    # plt.close(fig)
    cost_no_lockdown = posterior_cost.compute_cost(np.ones_like(mobility_initial), n_days, sigma, epsilon, backend)
    # fig, ax = posterior_cost.produce_plot(np.ones_like(mobility_initial), n_days)
    # plt.savefig(results_folder + filename_prefix + "evolution_under_no_lockdown.pdf")
    # plt.close(fig)
    print("Initial cost: {:.2f}, no-lockdown cost: {:.2f}".format(cost_initial, cost_no_lockdown))
    print(time.time() - t)

    # OPTIMAL CONTROL WITH NO MOVING WINDOW APPROACH
    if perform_standard_optimal_control:
        # bounds = different_bounds('startconstrained')
        bounds = different_bounds('realistic', n_days, mobility_initial, end_training_mobility_values)

        results_da = optimize.dual_annealing(posterior_cost.compute_cost, bounds=bounds,
                                             args=(n_days, sigma, epsilon, backend), maxiter=10, maxfun=1e3,
                                             x0=mobility_initial)
        # Plotting the figures
        mobility_initial = mobility_initial.reshape(3, n_days)  # 3 instead of 4 as we are not using alpha_home
        mobility_final = results_da.x.reshape(3, n_days)
        cost_final = posterior_cost.compute_cost(mobility_final, n_days, sigma, epsilon, backend)
        np.save(results_folder + filename_prefix + "mobility_standard", mobility_final)

    # MOVING WINDOW APPROACH
    if perform_iterative_strategy:
        print("Iterative strategy")
        # window_size = 30  # in days
        mobility_initial = copy.deepcopy(
            np.stack(
                (mobility_school[-1] * np.ones(shape=(window_size,)), mobility_work[-1] * np.ones(shape=(window_size,)),
                 mobility_other[-1] * np.ones(shape=(window_size,))))).flatten()

        # shift_each_iteration = 10  # number of days by which to shift the sliding window at each iteration.
        # n_shifts = 10
        total_days = n_shifts * shift_each_iteration
        print(total_days)

        total_mobility = np.zeros((3, total_days))

        if restart_at_index is not None:
            total_mobility = np.load(
                results_folder + filename_prefix + "mobility_iterative_" + str(restart_at_index) + ".npy")

        bounds = different_bounds('realistic', n_days=window_size, alpha_initial=mobility_initial,
                                  end_training_alpha_values=end_training_mobility_values)

        for shift_idx in range(n_shifts):
            print('Running shift: ' + str(shift_idx))
            if restart_at_index is not None and shift_idx <= restart_at_index:
                # we exploit the same loop in order to restart, so that the evolution of the model will be the same.
                mobility_final = np.zeros((3, window_size))
                mobility_final[:, 0:shift_each_iteration] = \
                    total_mobility[:, shift_idx * shift_each_iteration:(shift_idx + 1) * shift_each_iteration]
                # keep that constant for the future; this is only used to initialize the next optimal control iteration:
                mobility_final[:, shift_each_iteration:] = mobility_final[:, shift_each_iteration - 1].reshape(3, 1)
            else:
                # do the optimal control stuff
                results_da = optimize.dual_annealing(posterior_cost.compute_cost, bounds=bounds,
                                                     args=(window_size, sigma, epsilon, backend), maxiter=10,
                                                     maxfun=1e3, x0=mobility_initial)

                # get the result of the optimization in that time window
                mobility_final = results_da.x.reshape(3, window_size)
                # save it to the total_mobility array:
                total_mobility[:, shift_idx * shift_each_iteration:(shift_idx + 1) * shift_each_iteration] = \
                    mobility_final[:, 0:shift_each_iteration]
                # Save in between mobility steps
                np.save(results_folder + filename_prefix + "mobility_iterative_" + str(shift_idx), total_mobility)

            # update now the state of the model:
            posterior_cost.update_states(shift_each_iteration, mobility_final[:, :shift_each_iteration])

            # update mobility_initial as well, with the translated values of mobility_final, it may speed up convergence.
            mobility_initial_tmp = np.zeros_like(mobility_final)
            mobility_initial_tmp[:, 0:window_size - shift_each_iteration] = mobility_final[:,
                                                                            shift_each_iteration: window_size]
            mobility_initial_tmp[:, window_size - shift_each_iteration:] = np.stack(
                [mobility_final[:, window_size - shift_each_iteration - 1]] * shift_each_iteration, axis=1)
            mobility_initial = mobility_initial_tmp.flatten()

        np.save(results_folder + filename_prefix + "mobility_iterative", total_mobility)


def run_single_epsilon(epsilon, window_size=30, n_shifts=120, results_folder=None, journal_file_name=None,
                       training_window_length=None):
    main(epsilon=[epsilon, epsilon, epsilon], sigma=1, window_size=window_size,
         filename_prefix="mode_eps_{}_{}_{}_window_{}_Th_{}".format(epsilon, epsilon, epsilon, window_size, n_shifts),
         perform_iterative_strategy=True, perform_standard_optimal_control=False, shift_each_iteration=1,
         n_shifts=n_shifts, results_folder=results_folder, journal_file_name=journal_file_name,
         training_window_length=training_window_length)


def run_different_epsilon(epsilon_school, epsilon_work, epsilon_other, window_size=30, results_folder=None,
                          journal_file_name=None, training_window_length=None):
    main(epsilon=[epsilon_school, epsilon_work, epsilon_other], sigma=1, window_size=window_size,
         filename_prefix="mode_eps_{}_{}_{}_window_{}".format(epsilon_school, epsilon_work, epsilon_other, window_size),
         perform_iterative_strategy=True, perform_standard_optimal_control=False, shift_each_iteration=1, n_shifts=120,
         results_folder=results_folder, journal_file_name=journal_file_name,
         training_window_length=training_window_length)


def run_different_epsilon_with_post_mean(epsilon_school, epsilon_work, epsilon_other, window_size=30,
                                         results_folder=None, journal_file_name=None, training_window_length=None,
                                         n_post_samples=20, use_mpi=True, restart_at_index=None):
    main(epsilon=[epsilon_school, epsilon_work, epsilon_other], sigma=1, window_size=window_size,
         filename_prefix="mode_eps_{}_{}_{}_window_{}_post_samples_{}".format(epsilon_school, epsilon_work,
                                                                              epsilon_other, window_size,
                                                                              n_post_samples),
         perform_iterative_strategy=True, perform_standard_optimal_control=False, shift_each_iteration=1, n_shifts=120,
         results_folder=results_folder, journal_file_name=journal_file_name, use_sample_with_higher_weight=False,
         training_window_length=training_window_length, n_post_samples=n_post_samples, use_mpi=use_mpi,
         restart_at_index=restart_at_index)


def plot_different_epsilon_results(epsilon_school, epsilon_work, epsilon_other, window_size=30, results_folder=None,
                                   journal_file_name=None, training_window_length=None):
    main(epsilon=[epsilon_school, epsilon_work, epsilon_other], sigma=1, window_size=window_size,
         filename_prefix="optimal_control_results/mode_eps_{}_{}_{}_window_{}".format(epsilon_school, epsilon_work,
                                                                                      epsilon_other, window_size),
         results_folder=results_folder, journal_file_name=journal_file_name,
         training_window_length=training_window_length, only_plot=True,
         plot_file="optimal_control_results/mode_eps_{}_{}_{}_window_{}".format(epsilon_school, epsilon_work,
                                                                                epsilon_other,
                                                                                window_size) + "mobility_iterative.npy")


def plot_different_epsilon_results_post_mean(epsilon_school, epsilon_work, epsilon_other, window_size=30,
                                             results_folder=None, journal_file_name=None, training_window_length=None,
                                             n_post_samples=50, iteration=None, plot_days=120):
    main(epsilon=[epsilon_school, epsilon_work, epsilon_other], sigma=1, window_size=window_size,
         filename_prefix="mode_eps_{}_{}_{}_window_{}_post_samples_{}".format(epsilon_school,
                                                                                                      epsilon_work,
                                                                                                      epsilon_other,
                                                                                                      window_size,
                                                                                                      n_post_samples),
         results_folder=results_folder, journal_file_name=journal_file_name,
         training_window_length=training_window_length, only_plot=True, n_post_samples=n_post_samples,
         # plot_file="optimal_control_results/mode_eps_{}_{}_{}_window_{}".format(
         plot_file="optimal_control_results/mode_eps_{}_{}_{}_window_{}_post_samples_{}".format(
             epsilon_school, epsilon_work, epsilon_other, window_size, n_post_samples) + "mobility_iterative" + (
                       "" if iteration is None else "_{}".format(iteration)) + ".npy", plot_days=plot_days)


if __name__ == '__main__':
    results_folder = "results/SEI4RD_england_infer_1Mar_31Aug/"
    journal_file_name = "journal_3.jrl"
    training_window_length = None  # if this is None, it assumes that the trainin window is given by the full length of the mobility data.
    epsilon_school = 150
    epsilon_work = 300
    epsilon_other = 3

    run_different_epsilon_with_post_mean(epsilon_school=epsilon_school, epsilon_work=epsilon_work,
                                         epsilon_other=epsilon_other,
                                         results_folder=results_folder, journal_file_name=journal_file_name,
                                         training_window_length=training_window_length, n_post_samples=1, use_mpi=False,
                                         restart_at_index=None)
