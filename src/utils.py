from time import time

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import weighted
from abcpy.inferences import SABC, APMCABC, RejectionABC, SMCABC, ABCsubsim  # PMCABC
from matplotlib.cbook import violin_stats

from src.inferences_new import PMCABC


def ABC_inference(algorithm, model, observation, distance_calculator, eps, n_samples, n_steps, backend, seed=None,
                  full_output=0, kernel=None, **kwargs):
    start = time()
    if algorithm == "PMCABC":
        sampler = PMCABC([model], [distance_calculator], backend, seed=seed, kernel=kernel)
        jrnl = sampler.sample([[observation]], n_steps, np.array([eps]), n_samples=n_samples, full_output=full_output,
                              **kwargs)
    if algorithm == "APMCABC":
        sampler = APMCABC([model], [distance_calculator], backend, seed=seed, kernel=kernel)
        jrnl = sampler.sample([[observation]], n_steps, n_samples=n_samples, full_output=full_output, **kwargs)
    elif algorithm == "SABC":
        sampler = SABC([model], [distance_calculator], backend, seed=seed, kernel=kernel)
        jrnl = sampler.sample([[observation]], n_steps, eps, n_samples=n_samples, full_output=full_output, **kwargs)
    elif algorithm == "SMCABC":
        sampler = SMCABC([model], [distance_calculator], backend, seed=seed, kernel=kernel)
        jrnl = sampler.sample([[observation]], n_steps, n_samples=n_samples, full_output=full_output, **kwargs)
    elif algorithm == "RejectionABC":
        sampler = RejectionABC([model], [distance_calculator], backend, seed=seed)
        jrnl = sampler.sample([[observation]], epsilon=eps, n_samples=n_samples, n_samples_per_param=1,
                              full_output=full_output, **kwargs)
    elif algorithm == "ABCsubsim":
        sampler = ABCsubsim([model], [distance_calculator], backend, seed=seed, kernel=kernel)
        # chain_length=10, ap_change_cutoff=10
        jrnl = sampler.sample([[observation]], steps=n_steps, n_samples=n_samples, n_samples_per_param=1,
                              full_output=full_output, **kwargs)

    print("It took ", time() - start, " seconds.")
    return jrnl


def plot_results_model(res, start_step=0, end_step=None, show=True, fig=None, ax=None, **kwargs):
    susceptible_evolution, exposed_evolution, infected_c_evolution, infected_sc1_evolution, infected_sc2_evolution, \
    removed_evolution, deceased_evolution, confirmed_evolution = res[0]
    # pop_age_group = np.sum(
    #     [susceptible_evolution, exposed_evolution, infected_c_evolution, infected_sc_evolution, removed_evolution,
    #      deceased_evolution], axis=0)  # sanity check

    if end_step is None:
        end_step = susceptible_evolution.shape[-2]

    if fig is None or ax is None:
        fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(24, 8))
    ax[0, 0].plot(susceptible_evolution[start_step:end_step], **kwargs)
    ax[0, 1].plot(exposed_evolution[start_step:end_step], **kwargs)
    ax[0, 2].plot(infected_c_evolution[start_step:end_step], **kwargs)
    ax[0, 3].plot(infected_sc1_evolution[start_step:end_step], **kwargs)
    ax[1, 0].plot(infected_sc2_evolution[start_step:end_step], **kwargs)
    ax[1, 1].plot(removed_evolution[start_step:end_step], **kwargs)
    ax[1, 2].plot(deceased_evolution[start_step:end_step], **kwargs)
    ax[1, 3].plot(confirmed_evolution[start_step:end_step], **kwargs)
    ax[0, 0].set_title("Susceptible")
    ax[0, 1].set_title("Exposed")
    ax[0, 2].set_title("Infected clinical")
    ax[0, 3].set_title("Infected subclinical 1")
    ax[1, 0].set_title("Infected subclinical 2")
    ax[1, 1].set_title("Removed")
    ax[1, 2].set_title("Deceased")
    ax[1, 3].set_title("Confirmed cumulative")
    if show:
        plt.show()


def plot_confidence_bands(data, start_step=0, end_step=None, fig=None, ax=None, inner_band=50, outer_band=90,
                          alpha_median=1, alpha_inner=0.4, alpha_outer=0.2, color_inner="C1", color_outer="C1",
                          color_median="C3", fill_between=True, **kwargs):
    # data is a 2 dimensional array, in which the 1st dimension is the number of simulations and second is timestep.

    if end_step is None:
        end_step = data.shape[-1]

    median_simulation = np.median(data, axis=0)  # this takes the pointwise median; it is therefore not a trajectory
    lower_inner_simulation = np.percentile(data, 50 - inner_band / 2, axis=0)
    upper_inner_simulation = np.percentile(data, 50 + inner_band / 2, axis=0)
    lower_outer_simulation = np.percentile(data, 50 - outer_band / 2, axis=0)
    upper_outer_simulation = np.percentile(data, 50 + outer_band / 2, axis=0)

    if fig is None or ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))

    ax.plot(np.arange(start_step, end_step), median_simulation[start_step:end_step], color=color_median,
            alpha=alpha_median, **kwargs)
    if fill_between:
        ax.fill_between(np.arange(start_step, end_step), lower_inner_simulation[start_step:end_step],
                        upper_inner_simulation[start_step:end_step], alpha=alpha_inner, color=color_inner)
        ax.fill_between(np.arange(start_step, end_step), lower_outer_simulation[start_step:end_step],
                        upper_outer_simulation[start_step:end_step], alpha=alpha_outer, color=color_outer)
    else:
        ax.plot(np.arange(start_step, end_step), lower_inner_simulation[start_step:end_step],
                alpha=alpha_inner, color=color_inner, ls="--")
        ax.plot(np.arange(start_step, end_step),
                upper_inner_simulation[start_step:end_step], alpha=alpha_inner, color=color_inner, ls="--")
        ax.plot(np.arange(start_step, end_step), lower_outer_simulation[start_step:end_step],
                alpha=alpha_outer, color=color_outer, ls="--")
        ax.plot(np.arange(start_step, end_step), upper_outer_simulation[start_step:end_step], alpha=alpha_outer,
                color=color_outer, ls="--")


def plot_results_confidence_bands(data, start_step=0, end_step=None, show=True, fig=None, ax=None,
                                  color="C1", **kwargs):
    # data is a 3 dimensional array, in which the 1st dimension is the number of simulations, second is the trajectory type and third timestep.

    if end_step is None:
        end_step = data.shape[-1]

    if fig is None or ax is None:
        fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(24, 8))

    alpha_inner = 0.4
    alpha_outer = 0.2
    color_inner = color
    color_outer = color

    for i in range(4):
        plot_confidence_bands(data[:, i, :], start_step=start_step, end_step=end_step, fig=fig, ax=ax[0, i],
                              color_outer=color_outer, color_inner=color_inner, alpha_outer=alpha_outer,
                              alpha_inner=alpha_inner, **kwargs)
        plot_confidence_bands(data[:, i + 4, :], start_step=start_step, end_step=end_step, fig=fig, ax=ax[1, i],
                              color_outer=color_outer, color_inner=color_inner, alpha_outer=alpha_outer,
                              alpha_inner=alpha_inner, **kwargs)

    ax[0, 0].set_title("Susceptible")
    ax[0, 1].set_title("Exposed")
    ax[0, 2].set_title("Infected clinical")
    ax[0, 3].set_title("Infected subclinical 1")
    ax[1, 0].set_title("Infected subclinical 2")
    ax[1, 1].set_title("Removed")
    ax[1, 2].set_title("Deceased")
    ax[1, 3].set_title("Confirmed cumulative")
    if show:
        plt.show()


def generate_samples(model, k, seed=None, num_timeseries=5, two_dimensional=False):
    """Quite inefficient as it uses a loop to generate the k observations. It works also if one of the parameters of the model is an hyperparameter.
    This also works for parameters which are multidimensional. We use the two-dimensional flag in case the model returns already the observation."""

    if seed is not None:
        np.random.seed(seed)
    if not two_dimensional:
        if model.return_once_a_day:
            simulations = np.zeros((k, num_timeseries, model.T + 1, model.n_age_groups))
        else:
            simulations = np.zeros((k, num_timeseries, model.n_steps + 1, model.n_age_groups))

    else:
        if model.return_once_a_day:
            simulations = np.zeros((k, model.T + 1, num_timeseries))
        else:
            simulations = np.zeros((k, model.n_steps + 1, num_timeseries))

    parameters = np.zeros((k, model.total_num_params))

    for i in range(k):

        # we sample from the prior for the parameters of the model
        params = np.array([])
        for parent in model.get_input_models():
            if not parent.visited:
                # each parameter of the model is specified by a distribution with some fixed hyperparameters; the following gets these hyperparameters
                param_list_parent = []
                parent.visited = True
                for hyperparam in parent.get_input_models():
                    if not hyperparam.visited:
                        param_list_hyperparam = []
                        hyperparam.visited = True
                        for hyperparam2 in hyperparam.get_input_models():
                            if not hyperparam2.visited:
                                hyperparam2.visited = True
                                param_list_hyperparam.append(hyperparam2.forward_simulate([], 1)[0])
                        param_list_parent.append(hyperparam.forward_simulate(param_list_hyperparam, 1)[0])
                params = np.concatenate((params, parent.forward_simulate(param_list_parent, 1)[0]))

        parameters[i] = np.array(params).reshape(-1)
        simulations[i] = model.forward_simulate(parameters[i], 1)[0]

        # now reset the visited flag; this is not the most efficient way to do this operation.
        for parent in model.get_input_models():
            if parent.visited:
                parent.visited = False
                for hyperparam in parent.get_input_models():
                    if hyperparam.visited:
                        hyperparam.visited = False
                        for hyperparam2 in hyperparam.get_input_models():
                            if hyperparam2.visited:
                                hyperparam2.visited = False

    return parameters, simulations


def determine_eps(samples_matrix, dist_calc, quantile):
    dist = []
    for i in range(len(samples_matrix)):
        for j in range(i + 1, len(samples_matrix)):
            dist.append(dist_calc.distance([samples_matrix[i]], [samples_matrix[j]]))
    return np.quantile(dist, quantile)


def de_cumsum(x):
    z = np.zeros_like(x)
    z[0, :] = x[0, :]
    z[1:, :] = x[1:] - x[0:-1, :]
    return z


def stack_alphas(starting_alpha, alpha_3_timeseries, alpha_4_timeseries):
    if len(starting_alpha.shape) == 1:
        # print(starting_alpha.shape, alpha_3_timeseries.shape)
        alpha_new = np.stack(
            (starting_alpha, starting_alpha, starting_alpha, alpha_3_timeseries, alpha_4_timeseries),
            axis=1)
    elif len(starting_alpha.shape) == 2 and starting_alpha.shape[1] == 3:
        alpha_new = np.concatenate(
            (starting_alpha, alpha_3_timeseries.reshape(-1, 1), alpha_4_timeseries.reshape(-1, 1)), axis=1)
    else:
        raise RuntimeError
    return alpha_new


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.

    From https://stackoverflow.com/a/29677616/13516418
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def return_jrnl_credibility_interval(journal, percentile, iteration=None):
    above_quantile = (50 - percentile / 2.0) / 100
    below_quantile = (50 + percentile / 2.0) / 100

    weights = journal.get_weights(iteration).reshape(-1) / np.sum(journal.get_weights(iteration))
    params = journal.get_parameters(iteration)

    CI_dict = {}
    for key in params.keys():
        data = np.array(params[key]).reshape(-1)
        CI_dict[key] = weighted_quantile(data, [above_quantile, below_quantile], weights)

    return CI_dict


def improve_jrnl_posterior_plot(jrnl, parameter_names, titles_names, path_to_save,
                                figsize=None, label_size=None, title_size=None, ticks_size=None, **kwargs):
    """This works for the bivariate plot"""
    fig, axes = jrnl.plot_posterior_distr(parameter_names, **kwargs)

    if figsize is None:
        figsize = fig.get_size_inches() * fig.dpi  # size in pixels
    # set to default values if they are not provided
    if label_size is None:
        label_size = figsize / len(parameter_names) * 4
    if title_size is None:
        title_size = figsize / len(parameter_names) * 4.25
    if ticks_size is None:
        ticks_size = figsize / len(parameter_names) * 3

    if hasattr(axes, "shape"):
        for j, label in enumerate(parameter_names):
            axes[0, j].set_title(titles_names[j], size=title_size)

            if len(parameter_names) > 1:
                axes[j, 0].set_ylabel(titles_names[j], size=label_size)
                axes[-1, j].set_xlabel(titles_names[j], size=label_size)
            else:
                axes[j, 0].set_ylabel("Density", size=label_size)

            axes[j, 0].tick_params(axis='both', which='major', labelsize=ticks_size)
            axes[j, 0].ticklabel_format(style='plain', axis='both', scilimits=(0, 1e3))
            axes[j, 0].yaxis.offsetText.set_fontsize(ticks_size)

            axes[-1, j].tick_params(axis='both', which='major', labelsize=ticks_size)
            axes[-1, j].ticklabel_format(style='plain', axis='both', scilimits=(0, 1e3))
            axes[-1, j].xaxis.offsetText.set_fontsize(ticks_size)
            axes[j, -1].tick_params(axis='both', which='major', labelsize=ticks_size)
            axes[j, -1].ticklabel_format(style='plain', axis='both', scilimits=(0, 1e3))
            axes[j, -1].yaxis.offsetText.set_fontsize(ticks_size)

    elif isinstance(axes, list) and len(axes) == 1:
        axes[0].set_xlabel(titles_names[0], size=label_size)
        axes[0].set_ylabel(titles_names[1], size=label_size)
        axes[0].tick_params(axis='both', which='major', labelsize=ticks_size)
        axes[0].ticklabel_format(style='sci', axis='y', scilimits=(1e-3, 1e3))
        axes[0].xaxis.offsetText.set_fontsize(ticks_size)
        axes[0].yaxis.offsetText.set_fontsize(ticks_size)

    plt.savefig(path_to_save, bbox_inches="tight")
    plt.close(fig)
    return fig, axes


def vdensity_with_weights(weights):
    """ Taken from https://colab.research.google.com/drive/1cSnJGKJEqbllkPbF2z0cnfdwT40sUKKR#scrollTo=RIcLIr5XJRmx
    Outer function allows inner function access to weights. Matplotlib
    needs function to take in data and coords, so this seems like only way
    to 'pass' custom density function a set of weights """

    def vdensity(data, coords):
        ''' Custom matplotlib weighted violin stats function '''
        # Using weights from closure, get KDE fomr statsmodels
        weighted_cost = sm.nonparametric.KDEUnivariate(data)
        weighted_cost.fit(fft=False, weights=weights)

        # Return y-values for graph of KDE by evaluating on coords
        return weighted_cost.evaluate(coords)

    return vdensity


def custom_violin_stats(data, weights):
    """Taken from https://colab.research.google.com/drive/1cSnJGKJEqbllkPbF2z0cnfdwT40sUKKR#scrollTo=RIcLIr5XJRmx"""
    # Get weighted median and mean (using weighted module for median)
    median = weighted.quantile_1D(data, weights, 0.5)
    mean, sumw = np.ma.average(data, weights=list(weights), returned=True)

    # Use matplotlib violin_stats, which expects a function that takes in data and coords
    # which we get from closure above
    results = violin_stats(data, vdensity_with_weights(weights))

    # Update result dictionary with our updated info
    results[0][u"mean"] = mean
    results[0][u"median"] = median

    # No need to do this, since it should be populated from violin_stats
    # results[0][u"min"] =  np.min(data)
    # results[0][u"max"] =  np.max(data)

    return results


def violinplots(params_array, titles_names, weights, showmeans=True, showextrema=True, showmedians=True, **kwargs):
    fig, ax = plt.subplots(figsize=(1.2 * len(titles_names), 4))

    vpstats = []
    for i in range(params_array.shape[0]):
        vpstats.append(custom_violin_stats(params_array[i, :], weights.reshape(-1))[0])

    ax.violin(vpstats, vert=True, showmeans=showmeans, showextrema=showextrema, showmedians=showmedians, **kwargs)
    ax.set_xticks(np.arange(len(titles_names)) + 1)
    ax.set_xticklabels(titles_names)
    return fig, ax


def interleave_two_lists(l1, l2):
    return [val for pair in zip(l1, l2) for val in pair]
