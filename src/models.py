from functools import reduce

import numpy as np
from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector

from src.utils import stack_alphas, de_cumsum


class SEI4RD_abstract_model():
    """We implement here the epidemics evolution common to all variants"""

    def __init__(self):

        self.dt_sixth = self.dt / 6.0
        self.dt_half = 0.5 * self.dt

    def evolve_epidemics(self, n_steps, beta, kappa, gamma_c, gamma_r, gamma_rc, nu, rho, rho_prime, alpha_school,
                         alpha_work, alpha_home, alpha_other, return_clinical_categories=False):
        # define the arrays which will keep the evolution of the whole epidemics
        # if return_clinical_categories=True, we return 2 more trajectories for Ic1 and Ic2. 
        susceptible_evolution = np.zeros((n_steps + 1, self.susceptible_init.shape[0]))
        exposed_evolution = np.zeros((n_steps + 1, self.exposed_init.shape[0]))
        infected_c1_evolution = np.zeros((n_steps + 1, self.infected_c1_init.shape[0]))
        infected_c2_evolution = np.zeros((n_steps + 1, self.infected_c2_init.shape[0]))
        infected_sc1_evolution = np.zeros((n_steps + 1, self.infected_sc1_init.shape[0]))
        infected_sc2_evolution = np.zeros((n_steps + 1, self.infected_sc2_init.shape[0]))
        removed_evolution = np.zeros((n_steps + 1, self.removed_init.shape[0]))
        deceased_evolution = np.zeros((n_steps + 1, self.deceased_init.shape[0]))
        confirmed_evolution = np.zeros((n_steps + 1, self.infected_c1_init.shape[0]))

        susceptible_evolution[0] = self.susceptible_init
        exposed_evolution[0] = self.exposed_init
        infected_c1_evolution[0] = self.infected_c1_init
        infected_c2_evolution[0] = self.infected_c2_init
        infected_sc1_evolution[0] = self.infected_sc1_init
        infected_sc2_evolution[0] = self.infected_sc2_init
        removed_evolution[0] = self.removed_init
        deceased_evolution[0] = self.deceased_init
        confirmed_evolution[0] = self.confirmed_init

        # we want now to store the contact matrix over iterations, so that we can keep it and use it in backward step:

        if self.contact_matrix is not None:
            self.contact_matrix_evolution = self.contact_matrix
        else:
            self.contact_matrix_evolution = np.zeros((n_steps, *self.contact_matrix_work.shape))
            # this is slightly memory intensive, but not too much.

        for i in range(n_steps):
            # find the contact matrix at that timestep; it would be possible to do this only once if the alphas were not
            # learned

            # TODO here we waste computation as the contact matrix stays the same in a given day; we could optimize
            #  this.
            if len(self.contact_matrix_evolution.shape) == 2:
                # the contact matrix is fixed
                contact_matrix = self.contact_matrix_evolution
            else:  # compute it
                contact_matrix = self._get_contact_matrix(i, alpha_school, alpha_work, alpha_home, alpha_other)

                self.contact_matrix_evolution[i] = contact_matrix  # store the contact matrix at this iteration.

            susceptible_evolution[i + 1], exposed_evolution[i + 1], infected_sc1_evolution[i + 1], \
            infected_sc2_evolution[
                i + 1], infected_c1_evolution[i + 1], infected_c2_evolution[i + 1], removed_evolution[i + 1], \
            deceased_evolution[i + 1], confirmed_evolution[i + 1] = self._rk4step_forward_sei3rd(
                susceptible_evolution[i],
                exposed_evolution[i],
                infected_sc1_evolution[i],
                infected_sc2_evolution[i],
                infected_c1_evolution[i],
                infected_c2_evolution[i],
                removed_evolution[i],
                deceased_evolution[i],
                confirmed_evolution[i], beta, kappa,
                gamma_c, gamma_r, gamma_rc, nu, rho, rho_prime,
                contact_matrix)

        # in this way the output format does not change
        infected_c_evolution = infected_c1_evolution + infected_c2_evolution

        if return_clinical_categories:
            return susceptible_evolution, exposed_evolution, infected_c_evolution, infected_sc1_evolution, \
                   infected_sc2_evolution, removed_evolution, deceased_evolution, confirmed_evolution, \
                   infected_c1_evolution, infected_c2_evolution
        else:
            return susceptible_evolution, exposed_evolution, infected_c_evolution, infected_sc1_evolution, \
                   infected_sc2_evolution, removed_evolution, deceased_evolution, confirmed_evolution

    def evolve_epidemics_partial(self, n_steps, beta, kappa, gamma_c, gamma_r, gamma_rc, nu, rho, rho_prime,
                                 alpha_school, alpha_work, alpha_home, alpha_other):
        # define the arrays which will keep the evolution of the whole epidemics
        susceptible_evolution = np.zeros((n_steps + 1, self.susceptible_init.shape[0]))
        exposed_evolution = np.zeros((n_steps + 1, self.exposed_init.shape[0]))
        infected_c1_evolution = np.zeros((n_steps + 1, self.infected_c1_init.shape[0]))
        infected_c2_evolution = np.zeros((n_steps + 1, self.infected_c2_init.shape[0]))
        infected_sc1_evolution = np.zeros((n_steps + 1, self.infected_sc1_init.shape[0]))
        infected_sc2_evolution = np.zeros((n_steps + 1, self.infected_sc2_init.shape[0]))

        susceptible_evolution[0] = self.susceptible_init
        exposed_evolution[0] = self.exposed_init
        infected_c1_evolution[0] = self.infected_c1_init
        infected_c2_evolution[0] = self.infected_c2_init
        infected_sc1_evolution[0] = self.infected_sc1_init
        infected_sc2_evolution[0] = self.infected_sc2_init

        # we want now to store the contact matrix over iterations, so that we can keep it and use it in backward step:

        if self.contact_matrix is not None:
            self.contact_matrix_evolution = self.contact_matrix
        else:
            self.contact_matrix_evolution = np.zeros((n_steps, *self.contact_matrix_work.shape))
            # this is slightly memory intensive, but not too much.

        for i in range(n_steps):
            # find the contact matrix at that timestep; it would be possible to do this only once if the alphas were not
            # learned

            # TODO here we waste computation as the contact matrix stays the same in a given day; we could optimize
            #  this.
            if len(self.contact_matrix_evolution.shape) == 2:
                # the contact matrix is fixed
                contact_matrix = self.contact_matrix_evolution
            else:  # compute it
                contact_matrix = self._get_contact_matrix(i, alpha_school, alpha_work, alpha_home, alpha_other)

                self.contact_matrix_evolution[i] = contact_matrix  # store the contact matrix at this iteration.

            susceptible_evolution[i + 1], exposed_evolution[i + 1], infected_sc1_evolution[i + 1], \
            infected_sc2_evolution[
                i + 1], infected_c1_evolution[i + 1], infected_c2_evolution[
                i + 1] = self._rk4step_forward_sei3rd_partial(susceptible_evolution[i],
                                                              exposed_evolution[i],
                                                              infected_sc1_evolution[i],
                                                              infected_sc2_evolution[i],
                                                              infected_c1_evolution[i],
                                                              infected_c2_evolution[i],
                                                              beta, kappa, gamma_c, gamma_r, gamma_rc, nu, rho,
                                                              rho_prime,
                                                              contact_matrix)

        return susceptible_evolution, exposed_evolution, infected_sc1_evolution, infected_sc2_evolution, \
               infected_c1_evolution, infected_c2_evolution

    def evolve_epidemics_partial_sc(self, n_steps, beta, kappa, gamma_c, gamma_r, rho,
                                    alpha_school, alpha_work, alpha_home, alpha_other):
        # define the arrays which will keep the evolution of the whole epidemics
        susceptible_evolution = np.zeros((n_steps + 1, self.susceptible_init.shape[0]))
        exposed_evolution = np.zeros((n_steps + 1, self.exposed_init.shape[0]))
        infected_sc1_evolution = np.zeros((n_steps + 1, self.infected_sc1_init.shape[0]))
        infected_sc2_evolution = np.zeros((n_steps + 1, self.infected_sc2_init.shape[0]))

        susceptible_evolution[0] = self.susceptible_init
        exposed_evolution[0] = self.exposed_init
        infected_sc1_evolution[0] = self.infected_sc1_init
        infected_sc2_evolution[0] = self.infected_sc2_init

        # we want now to store the contact matrix over iterations, so that we can keep it and use it in backward step:

        if self.contact_matrix is not None:
            self.contact_matrix_evolution = self.contact_matrix
        else:
            self.contact_matrix_evolution = np.zeros((n_steps, *self.contact_matrix_work.shape))
            # this is slightly memory intensive, but not too much.

        for i in range(n_steps):
            # find the contact matrix at that timestep; it would be possible to do this only once if the alphas were not
            # learned

            # TODO here we waste computation as the contact matrix stays the same in a given day; we could optimize
            #  this.
            if len(self.contact_matrix_evolution.shape) == 2:
                # the contact matrix is fixed
                contact_matrix = self.contact_matrix_evolution
            else:  # compute it
                contact_matrix = self._get_contact_matrix(i, alpha_school, alpha_work, alpha_home, alpha_other)

                self.contact_matrix_evolution[i] = contact_matrix  # store the contact matrix at this iteration.

            susceptible_evolution[i + 1], exposed_evolution[i + 1], infected_sc1_evolution[i + 1], \
            infected_sc2_evolution[i + 1] = self._rk4step_forward_sei3rd_partial_sc(susceptible_evolution[i],
                                                                                    exposed_evolution[i],
                                                                                    infected_sc1_evolution[i],
                                                                                    infected_sc2_evolution[i],
                                                                                    beta, kappa, gamma_c, gamma_r,
                                                                                    rho, contact_matrix)

        return susceptible_evolution, exposed_evolution, infected_sc1_evolution, infected_sc2_evolution

    def _euler_timestep_update(self, susceptible, exposed, infected_sc1, infected_sc2, infected_c1, infected_c2,
                               removed, deceased, confirmed, beta, kappa, gamma_c, gamma_r, gamma_rc, nu, rho,
                               rho_prime, contact_matrix):
        """We start from a given state and update it according to the parameters, for one single step. This corresponds
        to Euler's method for integrating ODEs."""

        infected_sc = infected_sc1 + infected_sc2
        sus_to_exp = beta * susceptible * np.matmul(contact_matrix, infected_sc / self.tot_population) * self.dt
        exp_to_isc1 = rho * kappa * exposed * self.dt
        exp_to_isc2 = (1 - rho) * kappa * exposed * self.dt
        isc1_to_ic1 = rho_prime * gamma_c * infected_sc1 * self.dt
        isc1_to_ic2 = (1 - rho_prime) * gamma_c * infected_sc1 * self.dt
        isc2_to_rem = gamma_r * infected_sc2 * self.dt
        ic2_to_rem = gamma_rc * infected_c2 * self.dt
        ic1_to_dec = nu * infected_c1 * self.dt

        susceptible_next = susceptible - sus_to_exp
        exposed_next = exposed + sus_to_exp - exp_to_isc1 - exp_to_isc2
        infected_sc1_next = infected_sc1 + exp_to_isc1 - isc1_to_ic1 - isc1_to_ic2
        infected_sc2_next = infected_sc2 + exp_to_isc2 - isc2_to_rem
        infected_c1_next = infected_c1 + isc1_to_ic1 - ic1_to_dec
        infected_c2_next = infected_c2 - ic2_to_rem + isc1_to_ic2
        removed_next = removed + ic2_to_rem + isc2_to_rem
        deceased_next = deceased + ic1_to_dec
        confirmed_next = confirmed + isc1_to_ic1 + isc1_to_ic2

        return susceptible_next, exposed_next, infected_sc1_next, infected_sc2_next, infected_c1_next, \
               infected_c2_next, removed_next, deceased_next, confirmed_next

    def _rk4step_forward_sei3rd(self, susceptible, exposed, infected_sc1, infected_sc2, infected_c1, infected_c2,
                                removed, deceased, confirmed, beta, kappa, gamma_c, gamma_r, gamma_rc, nu, rho,
                                rho_prime, contact_matrix):

        y = np.array([susceptible, exposed, infected_sc1, infected_sc2, infected_c1, infected_c2, removed,
                      deceased, confirmed])

        fixed_args = np.array([beta, kappa, gamma_c, gamma_r, gamma_rc, nu, rho, rho_prime, contact_matrix])

        final_y = self._rk4step(self._sei3rd_ode, y, fixed_args)

        return [final_y[i] for i in range(final_y.shape[0])]

    def _rk4step_forward_sei3rd_partial(self, susceptible, exposed, infected_sc1, infected_sc2, infected_c1,
                                        infected_c2, beta, kappa, gamma_c, gamma_r,
                                        gamma_rc, nu, rho, rho_prime, contact_matrix):

        y = np.array([susceptible, exposed, infected_sc1, infected_sc2, infected_c1, infected_c2])

        fixed_args = np.array([beta, kappa, gamma_c, gamma_r, gamma_rc, nu, rho, rho_prime, contact_matrix])

        final_y = self._rk4step(self._sei3rd_ode_partial, y, fixed_args)

        return [final_y[i] for i in range(final_y.shape[0])]

    def _rk4step_forward_sei3rd_partial_sc(self, susceptible, exposed, infected_sc1, infected_sc2, beta, kappa, gamma_c,
                                           gamma_r, rho, contact_matrix):

        y = np.array([susceptible, exposed, infected_sc1, infected_sc2])

        fixed_args = np.array([beta, kappa, gamma_c, gamma_r, rho, contact_matrix])

        final_y = self._rk4step(self._sei3rd_ode_partial_sc, y, fixed_args)

        return [final_y[i] for i in range(final_y.shape[0])]

    def _sei3rd_ode(self, susceptible, exposed, infected_sc1, infected_sc2, infected_c1, infected_c2, removed,
                    deceased, confirmed, beta, kappa, gamma_c, gamma_r, gamma_rc, nu, rho, rho_prime, contact_matrix):
        # this evalutates the ode:
        infected_sc = infected_sc1 + infected_sc2
        sus_to_exp = beta * susceptible * np.matmul(contact_matrix, infected_sc / self.tot_population)
        exp_to_isc1 = rho * kappa * exposed
        exp_to_isc2 = (1 - rho) * kappa * exposed
        isc1_to_ic1 = rho_prime * gamma_c * infected_sc1
        isc1_to_ic2 = (1 - rho_prime) * gamma_c * infected_sc1
        isc2_to_rem = gamma_r * infected_sc2
        ic2_to_rem = gamma_rc * infected_c2
        ic1_to_dec = nu * infected_c1

        d_sus = - sus_to_exp
        d_exp = + sus_to_exp - exp_to_isc1 - exp_to_isc2
        d_isc1 = + exp_to_isc1 - isc1_to_ic1 - isc1_to_ic2
        d_isc2 = + exp_to_isc2 - isc2_to_rem
        d_ic1 = + isc1_to_ic1 - ic1_to_dec
        d_ic2 = - ic2_to_rem + isc1_to_ic2
        d_rem = + ic2_to_rem + isc2_to_rem
        d_dec = + ic1_to_dec
        d_con = + isc1_to_ic1 + isc1_to_ic2

        return [d_sus, d_exp, d_isc1, d_isc2, d_ic1, d_ic2, d_rem, d_dec, d_con]

    def _sei3rd_ode_partial(self, susceptible, exposed, infected_sc1, infected_sc2, infected_c1, infected_c2,
                            beta, kappa, gamma_c, gamma_r, gamma_rc, nu, rho, rho_prime, contact_matrix):
        # this evalutates the ode:
        infected_sc = infected_sc1 + infected_sc2
        sus_to_exp = beta * susceptible * np.matmul(contact_matrix, infected_sc / self.tot_population)
        exp_to_isc1 = rho * kappa * exposed
        exp_to_isc2 = (1 - rho) * kappa * exposed
        isc1_to_ic1 = rho_prime * gamma_c * infected_sc1
        isc1_to_ic2 = (1 - rho_prime) * gamma_c * infected_sc1
        isc2_to_rem = gamma_r * infected_sc2
        ic2_to_rem = gamma_rc * infected_c2
        ic1_to_dec = nu * infected_c1

        d_sus = - sus_to_exp
        d_exp = + sus_to_exp - exp_to_isc1 - exp_to_isc2
        d_isc1 = + exp_to_isc1 - isc1_to_ic1 - isc1_to_ic2
        d_isc2 = + exp_to_isc2 - isc2_to_rem
        d_ic1 = + isc1_to_ic1 - ic1_to_dec
        d_ic2 = - ic2_to_rem + isc1_to_ic2

        return [d_sus, d_exp, d_isc1, d_isc2, d_ic1, d_ic2]

    def _sei3rd_ode_partial_sc(self, susceptible, exposed, infected_sc1, infected_sc2,
                               beta, kappa, gamma_c, gamma_r, rho, contact_matrix):
        # this evalutates the ode:
        infected_sc = infected_sc1 + infected_sc2
        sus_to_exp = beta * susceptible * np.matmul(contact_matrix, infected_sc / self.tot_population)
        exp_to_isc1 = rho * kappa * exposed
        exp_to_isc2 = (1 - rho) * kappa * exposed
        isc1_to_ic = gamma_c * infected_sc1
        isc2_to_rem = gamma_r * infected_sc2

        d_sus = - sus_to_exp
        d_exp = + sus_to_exp - exp_to_isc1 - exp_to_isc2
        d_isc1 = + exp_to_isc1 - isc1_to_ic
        d_isc2 = + exp_to_isc2 - isc2_to_rem

        return [d_sus, d_exp, d_isc1, d_isc2]

    def compute_J_Ic(self, infected_c_evolution, phi_func):
        # cost of infections
        # assume mobility is a 3 x n_steps dimensional array, with first index being: (school, work, other); do not
        # consider the home part
        # the evolution vectors are n_steps x n_age_groups
        # phi is a function.
        return np.sum(phi_func(infected_c_evolution))

    def compute_J_mobility(self, mobility, epsilon):
        # this is the cost of the reduction of mobility
        return np.sum(epsilon * np.sum((1 - mobility) ** 2, axis=1))

    # general utilities:

    def _rk4step(self, ode, y, fixed_args):
        k1 = np.array(ode(*y, *fixed_args)).squeeze()
        k2 = np.array(ode(*(y + self.dt_half * k1), *fixed_args)).squeeze()
        k3 = np.array(ode(*(y + self.dt_half * k2), *fixed_args)).squeeze()
        k4 = np.array(ode(*(y + self.dt * k3), *fixed_args)).squeeze()

        return y + self.dt_sixth * (k1 + k4 + 2 * (k2 + k3))

    def set_state(self, susceptible_state, exposed_state, infected_c1_state, infected_c2_state, infected_sc1_state,
                  infected_sc2_state, removed_state, deceased_state, confirmed_state):
        # check here that the states have the correct size (each of them needs to be of size self.n_age_groups
        if not np.all(map(lambda x: hasattr(x, "shape") and len(x.shape) == 1 and x.shape[0] == self.n_age_groups,
                          [susceptible_state, exposed_state, infected_c1_state, infected_c2_state, infected_sc1_state,
                           infected_sc2_state, removed_state, deceased_state, confirmed_state])):
            raise RuntimeError("Incorrect initialization of the state variables.")

        self.susceptible_init = susceptible_state
        self.exposed_init = exposed_state
        self.infected_c1_init = infected_c1_state
        self.infected_c2_init = infected_c2_state
        self.infected_sc1_init = infected_sc1_state
        self.infected_sc2_init = infected_sc2_state
        self.removed_init = removed_state
        self.deceased_init = deceased_state
        self.confirmed_init = confirmed_state

    def generate_alphas_learn_old(self, alpha_123, alpha_4, alpha_5):

        alpha_123_timeseries = np.ones(self.n_steps + self.timesteps_every_day)
        alpha_123_timeseries[self.lockdown_timestep:] = alpha_123
        alpha_4_timeseries = np.ones(self.n_steps + self.timesteps_every_day)
        alpha_4_timeseries[self.lockdown_timestep:] = alpha_4
        alpha_5_timeseries = np.ones(self.n_steps + self.timesteps_every_day)
        alpha_5_timeseries[self.lockdown_timestep:] = alpha_5

        # now we stack with the other alphas; however, if alpha is a fixed scalar, do not apply the
        # transformation
        if not self.modify_alpha_school:
            alpha_school = self.alpha_school
        else:
            alpha_school = stack_alphas(alpha_123_timeseries * self.alpha_school, alpha_4_timeseries,
                                        alpha_5_timeseries)
        if not self.modify_alpha_work:
            alpha_work = self.alpha_work
        else:
            alpha_work = stack_alphas(alpha_123_timeseries * self.alpha_work, alpha_4_timeseries,
                                      alpha_5_timeseries)
        if not self.modify_alpha_home:
            alpha_home = self.alpha_home
        else:
            alpha_home = stack_alphas(alpha_123_timeseries * self.alpha_home, alpha_4_timeseries,
                                      alpha_5_timeseries)
        if not self.modify_alpha_other:
            alpha_other = self.alpha_other
        else:
            alpha_other = stack_alphas(alpha_123_timeseries * self.alpha_other, alpha_4_timeseries,
                                       alpha_5_timeseries)

        return alpha_school, alpha_work, alpha_home, alpha_other

    def _get_contact_matrix(self, index, alpha_school, alpha_work, alpha_home, alpha_other):
        alphas = map(lambda x: x if np.isscalar(x) else x[index], (alpha_school, alpha_work, alpha_home, alpha_other))
        products = map(  # if the alpha here is an array, it is a multiplier for each age group.
            lambda elem: elem[0] * elem[1] if np.isscalar(elem[0]) else np.einsum("i,ij->ij", elem[0], elem[1]),
            zip(alphas, (
                self.contact_matrix_school, self.contact_matrix_work, self.contact_matrix_home,
                self.contact_matrix_other)))
        return reduce(lambda a, b: a + b, products)


# NB: this now works for n_age_groups=5


class SEI4RD(ProbabilisticModel, Continuous, SEI4RD_abstract_model):

    def __init__(self, parameters, tot_population, T, contact_matrix_school, contact_matrix_work, contact_matrix_home,
                 contact_matrix_other, alpha_school=1, alpha_work=1, alpha_home=1, alpha_other=1,
                 modify_alpha_school=True, modify_alpha_work=True, modify_alpha_home=True, modify_alpha_other=True,
                 dt=0.1, return_once_a_day=False, return_observation_only_with_hospitalized=False,
                 learn_alphas_old=True, lockdown_day=20, name='SEI4RD'):
        """parameters contains 20 elements:
            beta
            d_L defining kappa
            d_C defining gamma_c
            d_R defining gamma_R
            d_RC defining gamma_RC
            d_D: defines nu
            rho_1: probability of a clinical infection for age group 1
            rho_2: probability of a clinical infection for age group 2
            rho_3: probability of a clinical infection for age group 3
            rho_4: probability of a clinical infection for age group 4
            rho_5: probability of a clinical infection for age group 5
            rho_prime_prime_1: probability of a clinical infection resulting in death for age group 1
            rho_prime_prime_2: probability of a clinical infection resulting in death for age group 2
            rho_prime_prime_3: probability of a clinical infection resulting in death for age group 3
            rho_prime_prime_4: probability of a clinical infection resulting in death for age group 4
            rho_prime_prime_5: probability of a clinical infection resulting in death for age group 5
            initial_exposed: the total number of infected individuals at the start of the dynamics, which is split in
                the different age groups and in exposed, Isc and Ic
            alpha_123: multiplying factor over the alphas for the different categories from google data, after
                lockdown_day for age groups 0, 1, 2
            alpha_4: alpha after lockdown_day for age group 3
            alpha_5: alpha after lockdown_day for age group 4

            Let's say n is the number of age ranges we consider.
            tot_population is an array of size n, which contains the total population for each age range.
            contact matrix needs to be n x n

            T is the time horizon of the observation. dt is the integration timestep. For now, simple Euler integration
            method is used.

            dt and T have to be measured with the same units of measure as d_L, d_I and so on (day usually)
            The contacts matrix represents the number of contact per unit time (in the correct unit of measure).

            The alphas can be scalars or arrays, and define the ratio of contact matrix which is active at each
            timestep. If all of them are scalars, then the total contact matrix is computed only once.

            If `learn_alphas_old` is True, then the alphas for the age groups 3 and 4 are assumed to be piecewise
            constant and the value after the lockdown day is learned. The lockdown day is denoted by the `lockdown_day`
            variable. In this case, alphas for the other age groups (at each day) are either an array for the 3 age
            groups or a scalar. Then, you need parameters `alpha_4`, `alpha_5` which describe dynamics alpha for the old
            age groups after lockdown_day.
            The `modify_alpha_*` parameters denote which of the alphas will be modified with the inferred values for the
             age groups.

            This model here is a variation on the original SEIcIscR model, in which every patient becomes Isc, then some
            go to Ic and others directly to R.
            """

        self.contact_matrix_school = contact_matrix_school
        self.contact_matrix_home = contact_matrix_home
        self.contact_matrix_work = contact_matrix_work
        self.contact_matrix_other = contact_matrix_other

        self.return_observation_only_with_hospitalized = return_observation_only_with_hospitalized
        self.learn_alphas_old = learn_alphas_old
        self.lockdown_day = lockdown_day

        self.modify_alpha_school = modify_alpha_school
        self.modify_alpha_work = modify_alpha_work
        self.modify_alpha_home = modify_alpha_home
        self.modify_alpha_other = modify_alpha_other

        if all(map(np.isscalar, (alpha_school, alpha_work, alpha_home, alpha_other))):
            self.contact_matrix = alpha_school * contact_matrix_school + alpha_work * contact_matrix_work + \
                                  alpha_home * contact_matrix_home + alpha_other * contact_matrix_other
        else:
            self.contact_matrix = None
            self.alpha_school = alpha_school
            self.alpha_home = alpha_home
            self.alpha_work = alpha_work
            self.alpha_other = alpha_other

        self.tot_population = tot_population
        self.n_steps = np.int(np.ceil(T / dt))
        self.T = T
        self.dt = dt
        self.timesteps_every_day = np.int(1 / dt)
        self.lockdown_timestep = self.timesteps_every_day * self.lockdown_day
        self.return_once_a_day = return_once_a_day

        self.n_age_groups = tot_population.shape[0]  # age groups.
        self.xs = np.arange(self.n_age_groups)

        self.total_num_params = 20

        if not isinstance(parameters, list):
            raise TypeError('Input of SEI4RD model is of type list')

        if len(parameters) != self.total_num_params:
            raise RuntimeError(
                'Input list must be of length {}, containing [beta, d_L, d_C, d_R, d_RC, d_D, rho_0, rho_1, rho_2, rho_3, '
                'rho_4, rho_prime_0, rho_prime_1, rho_prime_2, rho_prime_3, rho_prime_4, initial_exposed, alpha_123, '
                'alpha_4, alpha_5].'.format(
                    self.total_num_params))

        # initialize the different populations:
        self.infected_c1_init = np.zeros(self.n_age_groups)
        self.infected_c2_init = np.zeros(self.n_age_groups)
        self.infected_sc1_init = np.zeros(self.n_age_groups)
        self.infected_sc2_init = np.zeros(self.n_age_groups)
        self.removed_init = np.zeros(self.n_age_groups)
        self.deceased_init = np.zeros(self.n_age_groups)
        self.confirmed_init = np.zeros(self.n_age_groups)

        input_connector = InputConnector.from_list(parameters)
        ProbabilisticModel.__init__(self, input_connector, name)
        SEI4RD_abstract_model.__init__(self)  # need to call this

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != self.total_num_params:
            raise RuntimeError(
                'Input list must be of length {}, containing [beta, d_L, d_C, d_R, d_RC, d_D, rho_0, rho_1, rho_2, rho_3, '
                'rho_4, rho_prime_0, rho_prime_1, rho_prime_2, rho_prime_3, rho_prime_4, initial_exposed, alpha_123, '
                'alpha_4, alpha_5].'.format(self.total_num_params))

        # Check whether input is from correct domain
        beta = input_values[0]
        d_L = input_values[1]
        d_C = input_values[2]
        d_R = input_values[3]
        d_RC = input_values[4]
        d_D = input_values[5]
        p01 = input_values[6]
        p02 = input_values[7]
        p03 = input_values[8]
        p04 = input_values[9]
        p05 = input_values[10]
        p11 = input_values[11]
        p12 = input_values[12]
        p13 = input_values[13]
        p14 = input_values[14]
        p15 = input_values[15]
        initial_exposed_infected = input_values[16]
        alpha_123 = input_values[17]
        alpha_4 = input_values[18]
        alpha_5 = input_values[19]

        checks_successful = (0 <= beta <= 1) and (d_L >= 0) and (d_C >= 0) and (d_R >= 0) and (d_RC >= 0) and (
                0 <= p01 <= 1) and (0 <= p02 <= 1) and (0 <= p03 <= 1) and (0 <= p04 <= 1) and (0 <= p05 <= 1) and (
                                    0 <= p11 <= 1) and (0 <= p12 <= 1) and (0 <= p13 <= 1) and (0 <= p14 <= 1) and (
                                    0 <= p15 <= 1) and (0 <= alpha_123 <= 1) and (0 <= alpha_4 <= 1) and (
                                    0 <= alpha_5 <= 1) and (d_D >= 0) and (initial_exposed_infected >= 0)
        return checks_successful

    def _check_output(self, values):
        return True

    def get_output_dimension(self):
        # TODO modify this
        return 1

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), return_clinical_categories=False):
        beta = input_values[0]
        d_L = input_values[1]
        d_C = input_values[2]
        d_R = input_values[3]
        d_RC = input_values[4]
        d_D = input_values[5]
        p01 = input_values[6]
        p02 = input_values[7]
        p03 = input_values[8]
        p04 = input_values[9]
        p05 = input_values[10]
        p11 = input_values[11]
        p12 = input_values[12]
        p13 = input_values[13]
        p14 = input_values[14]
        p15 = input_values[15]
        initial_exposed_infected = input_values[16]
        alpha_123 = input_values[17]
        alpha_4 = input_values[18]
        alpha_5 = input_values[19]

        kappa = 1 / d_L
        gamma_c = 1 / d_C
        gamma_r = 1 / d_R
        gamma_rc = 1 / d_RC
        nu = 1 / d_D

        rho = np.array([p01, p02, p03, p04, p05]).reshape(5, )
        rho_prime = np.array([p11, p12, p13, p14, p15]).reshape(5, )

        # split the number of initially exposed/infected people into the different age groups
        # 1/3 rd of them to exposed and 2/3 rd of to infectious and in different age groups according to rho
        initial_exposed_infected = np.array([0.1, 0.4, 0.35, 0.1, 0.05]) * initial_exposed_infected
        initial_exposed_age_groups = (1 / 3) * initial_exposed_infected
        initial_sc1_age_groups = rho * (2 / 3) * initial_exposed_infected
        initial_sc2_age_groups = (1 - rho) * (2 / 3) * initial_exposed_infected
        self.susceptible_init = self.tot_population - initial_exposed_age_groups - initial_sc1_age_groups - \
                                initial_sc2_age_groups
        self.exposed_init = initial_exposed_age_groups
        self.infected_sc1_init = initial_sc1_age_groups
        self.infected_sc2_init = initial_sc2_age_groups

        results = []

        # Do the actual forward simulation
        for i in range(k):
            # define now the correct alpha vectors, with the alpha_4 and alpha_5
            if self.learn_alphas_old:
                # we rearrange the things:
                alpha_school, alpha_work, alpha_home, alpha_other = self.generate_alphas_learn_old(alpha_123, alpha_4,
                                                                                                   alpha_5)

            else:
                alpha_school = self.alpha_school
                alpha_work = self.alpha_work
                alpha_home = self.alpha_home
                alpha_other = self.alpha_other

            results.append(
                self.evolve_epidemics(self.n_steps, beta, kappa, gamma_c, gamma_r, gamma_rc, nu, rho, rho_prime,
                                      alpha_school=alpha_school, alpha_work=alpha_work,
                                      alpha_home=alpha_home, alpha_other=alpha_other,
                                      return_clinical_categories=return_clinical_categories))

        # adapt to the API, ie transform each simulation outcome in an array.
        results = [np.array([x], copy=True) for x in results]
        results = [x.reshape(x.shape[1:]) for x in results]

        if self.return_once_a_day:
            results = [x[:, ::self.timesteps_every_day, :] for x in results]

        if self.return_observation_only_with_hospitalized:
            # extract the correct timeseries (ie the number of deceased and the Ic; we also de-cumsum (extract the daily
            # increments) for the deceased population
            results = [np.concatenate((de_cumsum(x[6]), np.sum(x[2], axis=1).reshape(-1, 1)), axis=1) for x in
                       results]

        return results

    def compute_R_evolution_with_actual_susceptible_diff_API(self, input_values, susceptible_evolution,
                                                             use_stored_contact_matrix=True):
        """This uses the contact matrix that has been computed and stored in the evolution of the epidemics method.
         Then it should be called immediately after forward_simulate or evolve_epidemics methods."""
        beta = input_values[0]
        d_L = input_values[1]
        d_C = input_values[2]
        d_R = input_values[3]
        d_RC = input_values[4]
        d_D = input_values[5]
        p01 = input_values[6]
        p02 = input_values[7]
        p03 = input_values[8]
        p04 = input_values[9]
        p05 = input_values[10]
        p11 = input_values[11]
        p12 = input_values[12]
        p13 = input_values[13]
        p14 = input_values[14]
        p15 = input_values[15]

        kappa = 1 / d_L
        gamma_c = 1 / d_C
        gamma_r = 1 / d_R
        gamma_rc = 1 / d_RC
        nu = 1 / d_D

        rho = np.array([p01, p02, p03, p04, p05]).reshape(5, )
        rho_prime = np.array([p11, p12, p13, p14, p15]).reshape(5, )
        return self.compute_R_with_actual_susceptible_evolution(beta, kappa, rho, rho_prime, gamma_c, gamma_r, gamma_rc,
                                                                nu,
                                                                susceptible_evolution,
                                                                use_stored_contact_matrix=use_stored_contact_matrix)

    def compute_R_with_actual_susceptible_evolution(self, beta, kappa, rho, rho_prime, gamma_c, gamma_r, gamma_rc, nu,
                                                    susceptible_evolution,
                                                    use_stored_contact_matrix=True):
        """ NB: This uses the contact matrix that has been computed and stored in the evolution of the epidemics method.
         Then it should be called immediately after forward_simulate or evolve_epidemics methods."""

        R = np.zeros(susceptible_evolution.shape[0] - 1)
        # discard the last element, as we have no contact matrix for it:
        for i in range(susceptible_evolution.shape[0] - 1):
            R[i] = self.compute_R_with_actual_susceptible(susceptible_evolution[i], beta, kappa, rho, rho_prime,
                                                          gamma_c, gamma_r, gamma_rc, nu,
                                                          self.contact_matrix_evolution[i * self.timesteps_every_day])
        return R

    def compute_R_with_actual_susceptible(self, susceptible, beta, kappa, rho, rho_prime, gamma_C, gamma_R, gamma_RC,
                                          nu, contact_matrix):
        A_matrix = np.zeros((30, 30))  # the matrix does not consider states for R, D

        term = beta * np.einsum("i,ij,j->ij", susceptible, contact_matrix, 1 / self.tot_population)
        A_matrix[0:5, 10:15] = -term
        A_matrix[0:5, 15:20] = -term
        A_matrix[5:10, 10:15] = term
        A_matrix[5:10, 15:20] = term

        A_matrix[5:10, 5:10] = -kappa * np.eye(5)
        A_matrix[10:15, 5:10] = kappa * np.diag(rho)
        A_matrix[10: 15, 10: 15] = -gamma_C * np.eye(5)
        A_matrix[15: 20, 5: 10] = kappa * np.diag(1 - rho)
        A_matrix[15: 20, 15: 20] = -gamma_R * np.eye(5)
        A_matrix[20: 25, 10: 15] = gamma_C * np.diag(rho_prime)
        A_matrix[20: 25, 20: 25] = -nu * np.eye(5)
        A_matrix[25: 30, 10: 15] = gamma_C * np.diag(1 - rho_prime)
        A_matrix[25: 30, 25: 30] = -gamma_RC * np.eye(5)

        T_matrix = np.zeros((25, 25))
        T_matrix[0:5, 5:] = A_matrix[5:10, 10:]
        Sigma_matrix = np.zeros((25, 25))
        Sigma_matrix[0:5, 0:5] = A_matrix[5:10, 5:10]
        Sigma_matrix[5:, :] = A_matrix[10:, 5:]

        K_L = - np.matmul(T_matrix, np.linalg.inv(Sigma_matrix))
        return np.max(np.linalg.eigvals(K_L))

    def compute_R_evolution_diff_API(self, input_values, use_stored_contact_matrix=True):
        """This uses the contact matrix that has been computed and stored in the evolution of the epidemics method.
         Then it should be called immediately after forward_simulate or evolve_epidemics methods."""
        beta = input_values[0]
        d_L = input_values[1]
        d_C = input_values[2]
        d_R = input_values[3]
        d_RC = input_values[4]
        d_D = input_values[5]
        p01 = input_values[6]
        p02 = input_values[7]
        p03 = input_values[8]
        p04 = input_values[9]
        p05 = input_values[10]
        p11 = input_values[11]
        p12 = input_values[12]
        p13 = input_values[13]
        p14 = input_values[14]
        p15 = input_values[15]

        kappa = 1 / d_L
        gamma_c = 1 / d_C
        gamma_r = 1 / d_R
        gamma_rc = 1 / d_RC
        nu = 1 / d_D

        rho = np.array([p01, p02, p03, p04, p05]).reshape(5, )
        rho_prime = np.array([p11, p12, p13, p14, p15]).reshape(5, )
        return self.compute_R_evolution(beta, kappa, rho, rho_prime, gamma_c, gamma_r, gamma_rc, nu,
                                        use_stored_contact_matrix=use_stored_contact_matrix)

    def compute_R_evolution(self, beta, kappa, rho, rho_prime, gamma_c, gamma_r, gamma_rc, nu,
                            use_stored_contact_matrix=True):
        """ NB: This uses the contact matrix that has been computed and stored in the evolution of the epidemics method.
         Then it should be called immediately after forward_simulate or evolve_epidemics methods."""

        R = np.zeros(self.contact_matrix_evolution[::self.timesteps_every_day].shape[0])
        # discard the last element, as we have no contact matrix for it:
        for i in range(self.contact_matrix_evolution[::self.timesteps_every_day].shape[0]):
            R[i] = self.compute_R(beta, kappa, rho, rho_prime, gamma_c, gamma_r, gamma_rc, nu,
                                  self.contact_matrix_evolution[i * self.timesteps_every_day])
        return R

    def compute_R(self, beta, kappa, rho, rho_prime, gamma_C, gamma_R, gamma_RC, nu, contact_matrix):
        """This assumes all population is susceptible"""
        A_matrix = np.zeros((30, 30))  # the matrix does not consider states for R, D

        term = beta * np.einsum("i,ij,j->ij", self.tot_population, contact_matrix, 1 / self.tot_population)
        A_matrix[0:5, 10:15] = -term
        A_matrix[0:5, 15:20] = -term
        A_matrix[5:10, 10:15] = term
        A_matrix[5:10, 15:20] = term

        A_matrix[5:10, 5:10] = -kappa * np.eye(5)
        A_matrix[10:15, 5:10] = kappa * np.diag(rho)
        A_matrix[10: 15, 10: 15] = -gamma_C * np.eye(5)
        A_matrix[15: 20, 5: 10] = kappa * np.diag(1 - rho)
        A_matrix[15: 20, 15: 20] = -gamma_R * np.eye(5)
        A_matrix[20: 25, 10: 15] = gamma_C * np.diag(rho_prime)
        A_matrix[20: 25, 20: 25] = -nu * np.eye(5)
        A_matrix[25: 30, 10: 15] = gamma_C * np.diag(1 - rho_prime)
        A_matrix[25: 30, 25: 30] = -gamma_RC * np.eye(5)

        T_matrix = np.zeros((25, 25))
        T_matrix[0:5, 5:] = A_matrix[5:10, 10:]
        Sigma_matrix = np.zeros((25, 25))
        Sigma_matrix[0:5, 0:5] = A_matrix[5:10, 5:10]
        Sigma_matrix[5:, :] = A_matrix[10:, 5:]

        K_L = - np.matmul(T_matrix, np.linalg.inv(Sigma_matrix))
        return np.max(np.linalg.eigvals(K_L))
