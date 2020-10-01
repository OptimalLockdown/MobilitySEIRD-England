import matplotlib.pyplot as plt
import pandas as pd
from abcpy.continuousmodels import Uniform
from abcpy.output import Journal
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from sklearn.utils.extmath import weighted_mode

from src.distance import *
from src.models import SEI4RD
from src.utils import plot_confidence_bands, plot_results_model, return_jrnl_credibility_interval, weighted_quantile, \
    improve_jrnl_posterior_plot, violinplots, interleave_two_lists

# date_to_use = '11/04'
# date_to_use = '26/04'
# date_to_use = '11/05'
# date_to_use = '23/05'
date_to_use = '31/08'
print("End training date:", date_to_use)

# from time import sleep
# sleep(60)
# DATA AND RESULTS:
#training_data_folder = "data/england_inference_data_1Mar_to_23May/"
#most_recent_data_folder = "data/england_inference_data_1Mar_to_23May/"
training_data_folder = "data/england_inference_data_1Mar_to_31Aug/"
most_recent_data_folder = "data/england_inference_data_1Mar_to_31Aug/"

if date_to_use == '11/04':
    journal_folder = "results/SEI4RD_england_infer_1Mar_11Apr/"
    images_folder = "results/SEI4RD_england_infer_1Mar_11Apr/"
    jrnl = Journal.fromFile(journal_folder + "journal_3_1.jrl")
    T_training = 42  # 11th Apr
if date_to_use == '26/04':
    journal_folder = "results/SEI4RD_england_infer_1Mar_26Apr/"
    images_folder = "results/SEI4RD_england_infer_1Mar_26Apr/"
    jrnl = Journal.fromFile(journal_folder + "journal_3.jrl")
    T_training = 57  # 26th Apr
elif date_to_use == '11/05':
    journal_folder = "results/SEI4RD_england_infer_1Mar_11May/"
    images_folder = "results/SEI4RD_england_infer_1Mar_11May/"
    jrnl = Journal.fromFile(journal_folder + "journal_3.jrl")
    T_training = 72  # 11th May
elif date_to_use == '23/05':
    journal_folder = "results/SEI4RD_england_infer_1Mar_23May/"
    images_folder = "results/SEI4RD_england_infer_1Mar_23May/"
    jrnl = Journal.fromFile(journal_folder + "PMCABC_inf3.jrl")
    mobility_home = np.load(training_data_folder + "mobility_home.npy")
    T_training = mobility_home.shape[0]
elif date_to_use == '31/08':
    journal_folder = "results/SEI4RD_england_infer_1Mar_31Aug/"
    images_folder = "results/SEI4RD_england_infer_1Mar_31Aug/"
    jrnl = Journal.fromFile(journal_folder + "journal_3.jrl")
    mobility_home = np.load(training_data_folder + "mobility_home.npy")
    T_training = mobility_home.shape[0]

print("Post mean:", jrnl.posterior_mean())
print("Params post mean {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.0f} & {:.2f} & {:.2f} & {:.2f}".format(
    *[jrnl.posterior_mean()[name] for name in
      ["beta", "d_L", "d_C", "d_R", "d_RC", "d_D", "initial_exposed", "alpha_123", "alpha_4", "alpha_5"]]))
print("rho post mean {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
    *[jrnl.posterior_mean()[name] for name in ["p01", "p02", "p03", "p04", "p05"]]))
print("rho_prime post mean {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
    *[jrnl.posterior_mean()[name] for name in ["p11", "p12", "p13", "p14", "p15"]]))

posterior_var = np.diag(jrnl.posterior_cov()[0])
dict_keys = jrnl.posterior_cov()[1]
posterior_std_dict = {key: np.sqrt(posterior_var[i]) for i, key in enumerate(dict_keys)}

print("Post mean:", jrnl.posterior_mean())
print("Days")
print(
    "${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$".format(
        *interleave_two_lists([jrnl.posterior_mean()[name] for name in ["d_L", "d_C", "d_R", "d_RC", "d_D", ]],
                              [posterior_std_dict[name] for name in ["d_L", "d_C", "d_R", "d_RC", "d_D", ]])))
print("Beta alphas")
print(
    "${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.0f} \pm {:.0f}$".format(
        *interleave_two_lists(
            [jrnl.posterior_mean()[name] for name in ["beta", "alpha_123", "alpha_4", "alpha_5", "initial_exposed", ]],
            [posterior_std_dict[name] for name in ["beta", "alpha_123", "alpha_4", "alpha_5", "initial_exposed", ]])))
print("rhos")
print(
    # "[{:.2f} \pm {:.2f}, {:.2f} \pm {:.2f}, {:.2f} \pm {:.2f}, {:.2f} \pm {:.2f}, {:.2f} \pm {:.2f}]".format(
    "${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$".format(
        *interleave_two_lists([jrnl.posterior_mean()[name] for name in ["p01", "p02", "p03", "p04", "p05"]],
                              [posterior_std_dict[name] for name in ["p01", "p02", "p03", "p04", "p05"]])))
print("rho primes")
print(
    # "[{:.2f} \pm {:.2f}, {:.2f} \pm {:.2f}, {:.2f} \pm {:.2f}, {:.2f} \pm {:.2f}, {:.2f} \pm {:.2f}]".format(
    "${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$".format(
        *interleave_two_lists([jrnl.posterior_mean()[name] for name in ["p11", "p12", "p13", "p14", "p15"]],
                              [posterior_std_dict[name] for name in ["p11", "p12", "p13", "p14", "p15"]])))

print(return_jrnl_credibility_interval(jrnl, 95))
# LOAD DATA
# mobility_home = np.load(most_recent_data_folder + "mobility_home.npy")
alpha_home = 1
mobility_work = np.load(most_recent_data_folder + "mobility_work.npy")
mobility_other = np.load(most_recent_data_folder + "mobility_other.npy")
mobility_school = np.load(most_recent_data_folder + "mobility_school.npy")

england_pop = np.load(most_recent_data_folder + "england_pop.npy", allow_pickle=True)

contact_matrix_home_england = np.load(most_recent_data_folder + "contact_matrix_home_england.npy")
contact_matrix_work_england = np.load(most_recent_data_folder + "contact_matrix_work_england.npy")
contact_matrix_school_england = np.load(most_recent_data_folder + "contact_matrix_school_england.npy")
contact_matrix_other_england = np.load(most_recent_data_folder + "contact_matrix_other_england.npy")

observation_england_more_recent = np.load(most_recent_data_folder + "observed_data.npy", allow_pickle=True)
T_most_recent_data = mobility_school.shape[0]

print(T_training, T_most_recent_data)
# SETTINGS:
iteration = -1  # iteration to use for plots (only works if full_output=1)
n_post_samples = 100  # to be used for final confidence bands plots.
# start and final day for the trajectories plots:
start_step = 19
if date_to_use == '11/04':
    # end_step = 60  # date until which to show predictions; for 11th Apr
    # end_step_observation = 60  # date until which to show the actual values; for 11th Apr
    end_step = T_most_recent_data + 7  # date until which to show predictions
    end_step_observation = T_most_recent_data  # date until which to show the actual values
if date_to_use == '26/04':
    end_step = T_most_recent_data + 7  # date until which to show predictions
    end_step_observation = T_most_recent_data  # date until which to show the actual values
elif date_to_use == '11/05':
    end_step = T_most_recent_data + 7  # date until which to show predictions
    end_step_observation = T_most_recent_data  # date until which to show the actual values
elif date_to_use == '23/05':
    end_step = T_most_recent_data + 7  # date until which to show predictions
    end_step_observation = T_most_recent_data  # date until which to show the actual values
elif date_to_use == '31/08':
    end_step = T_most_recent_data + 7  # date until which to show predictions
    end_step_observation = T_most_recent_data - 8  # date until which to show the actual values
return_observation_only_hospitalized = True
plot_posteriors_splitted = True
plot_posteriors_rhos = False
plot_posteriors_all = False
plot_posteriors_marginals = False
plot_violinplots = False
plot_trajectories = True
plot_posterior_mean_traj = False
CI_size = 99
seed = 1
np.random.seed(seed)

# DEFINE MODEL:

# parameters
n = 5  # number of age groups
dt = 0.1  # integration timestep
T = T_most_recent_data + 21  # horizon time in days (needs to be 1 less than the number of days in observation for
# T = T_most_recent_data + 40  # horizon time in days (needs to be 1 less than the number of days in observation for
# inference)
total_population = england_pop  # population for each age group
# 16th March: Boris Johnson asked old people to isolate; we then learn a new alpha from the 18th March:
lockdown_day = 17
fake_observation = False
learn_alphas = True

# keep last values of alpha for future predictions:
# alpha_home = np.concatenate((alpha_home, np.zeros((T - T_most_recent_data + 1, *alpha_home.shape[1:]))))
mobility_work = np.concatenate((mobility_work, np.zeros((T - T_most_recent_data + 1, *mobility_work.shape[1:]))))
mobility_other = np.concatenate((mobility_other, np.zeros((T - T_most_recent_data + 1, *mobility_other.shape[1:]))))
mobility_school = np.concatenate((mobility_school, np.zeros((T - T_most_recent_data + 1, *mobility_school.shape[1:]))))

# alpha_home[T_most_recent_data:] = alpha_home[T_most_recent_data - 1]
mobility_work[T_most_recent_data:] = mobility_work[T_most_recent_data - 1]
mobility_other[T_most_recent_data:] = mobility_other[T_most_recent_data - 1]
mobility_school[T_most_recent_data:] = mobility_school[T_most_recent_data - 1]

# alpha_home = np.repeat(alpha_home, np.int(1 / dt), axis=0)
mobility_work = np.repeat(mobility_work, np.int(1 / dt), axis=0)
mobility_other = np.repeat(mobility_other, np.int(1 / dt), axis=0)
mobility_school = np.repeat(mobility_school, np.int(1 / dt), axis=0)

# ABC model (priors need to be fixed better):
beta = Uniform([[0], [0.5]], name='beta')  # controls how fast the epidemics grows. Related to R_0
d_L = Uniform([[1], [16]], name='d_L')  # average duration of incubation
d_C = Uniform([[1], [16]], name='d_C')  # average time before going to clinical
d_R = Uniform([[1], [16]], name='d_R')  # average recovery time
d_RC = Uniform([[1], [16]], name='d_RC')  # average recovery time from clinical state
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

model_variables = [beta, d_L, d_C, d_R, d_RC, d_D, p01, p02, p03, p04, p05, p11, p12, p13, p14, p15, initial_exposed,
                   alpha_123, alpha_4, alpha_5]
model_keys = ["beta", "d_L", "d_C", "d_R", "d_RC", "d_D", "p01", "p02", "p03", "p04", "p05", 'p11', 'p12', 'p13', 'p14',
              'p15',
              "initial_exposed", "alpha_123", "alpha_4", "alpha_5"]
model_variables_dict = {key: value for (key, value) in zip(model_keys, model_variables)}

model = SEI4RD(model_variables, tot_population=total_population, T=T,
               contact_matrix_school=contact_matrix_school_england, contact_matrix_work=contact_matrix_work_england,
               contact_matrix_home=contact_matrix_home_england, contact_matrix_other=contact_matrix_other_england,
               alpha_school=mobility_school, alpha_work=mobility_work, alpha_home=alpha_home,
               alpha_other=mobility_other,
               modify_alpha_home=False, dt=dt, return_once_a_day=True,
               return_observation_only_with_hospitalized=return_observation_only_hospitalized, learn_alphas_old=True,
               lockdown_day=lockdown_day)

# if fake_observation:
#     true_parameters_fake = [0.05, 5, 7, 5, 6, 0.2, 4, np.pi * 1 / 8, 4.5, 50, 4, 0, 0.3, 0.3]
#     true_parameters_fake_dict = {key: value for (key, value) in zip(model_keys, true_parameters_fake)}
#     observation_england_more_recent = model.forward_simulate(true_parameters_fake, 1)
#
# PRODUCE EPSILON PLOT:
if "epsilon_arr" in jrnl.configuration.keys():
    print("Produce epsilon plot")
    # plt.plot(np.arange(1, len(jrnl.configuration["epsilon_arr"]) + 1), jrnl.configuration["epsilon_arr"], "o")
    plt.semilogy(np.arange(1, len(jrnl.configuration["epsilon_arr"]) + 1),
                 jrnl.configuration["epsilon_arr"], "o")
    plt.xlabel("Step")
    plt.ylabel(r"$\epsilon$")
    plt.savefig(images_folder + "epsilons.pdf")
    plt.close()

# PRODUCE DISTANCES PLOT:
dist_min = np.array([np.min(jrnl.distances[i]) for i in range(len(jrnl.distances))])
plt.plot(np.arange(1, len(jrnl.distances) + 1), dist_min, ".")
plt.xlabel("Step")
plt.ylabel("Min distance")
plt.savefig(images_folder + "distances.pdf")
plt.close()

# ESS:
print(len(jrnl.number_of_simulations))
for i in range(len(jrnl.number_of_simulations)):
    print("{:.2f}, \t {}, \t {:.2f}".format(
        1 / sum(pow(jrnl.get_weights(i) / sum(jrnl.get_weights(i)), 2))[0],
        jrnl.number_of_simulations[i], np.mean(jrnl.distances[i])))

# extract stuff
weights = jrnl.get_weights(iteration) / np.sum(jrnl.get_weights(iteration))
params = jrnl.get_parameters(iteration)
params_array = np.array([[params[key][i] for i in range(len(params[key]))] for key in params.keys()]).squeeze()
# temporary way to find the mode; it does not work well.
mode_vals, score = weighted_mode(params_array, weights.squeeze(), axis=1)
marginal_mode = {key: weighted_mode(np.array(params[key]).reshape(-1), weights.squeeze())[0] for key in params.keys()}
print("MAP:", marginal_mode)
marginal_medians = {key: weighted_quantile(np.array(params[key]).reshape(-1), [0.5], weights.squeeze()) for key in
                    params.keys()}
print(marginal_medians)

# PRODUCE POSTERIOR PLOTS:
param_list = ["beta", "d_L", "d_C", "d_R", "d_RC", "d_D", "p01", "p02", "p03", "p04", "p05", "p11", "p12", "p13", "p14",
              "p15", "initial_exposed"]
param_titles_list = [r"$\beta$", r"$d_L$", r"$d_C$", r"$d_R$", r"$d_{RC}$", r"$d_D$", r"$\rho_1$", r"$\rho_2$",
                     r"$\rho_3$", r"$\rho_4$", r"$\rho_5$", r"$\rho'_1$", r"$\rho'_2$", r"$\rho'_3$", r"$\rho'_4$",
                     r"$\rho'_5$", r"$N^{in}$"]
if learn_alphas:
    param_list += ["alpha_123", "alpha_4", "alpha_5"]
    param_titles_list += [r"$\alpha_{123}$", r"$\alpha_4$", r"$\alpha_5$"]

label_size = 20
title_size = 20
ticks_size = 13
if plot_posteriors_splitted:
    ticks_size = 13
    improve_jrnl_posterior_plot(jrnl, ["beta", "d_L", "d_C", "d_R", "d_RC", "d_D", "initial_exposed"],
                                param_titles_list[0:6] + [param_titles_list[-4]], label_size=label_size,
                                title_size=title_size, path_to_save=images_folder + "posterior1.pdf",
                                show_density_values=False, ticks_size=ticks_size)
    improve_jrnl_posterior_plot(jrnl, param_list[6:11], param_titles_list[6:11], label_size=label_size,
                                title_size=title_size, ticks_size=ticks_size,
                                path_to_save=images_folder + "posterior2.pdf",
                                show_density_values=False)
    improve_jrnl_posterior_plot(jrnl, param_list[11:16], param_titles_list[11:16], label_size=label_size,
                                title_size=title_size, ticks_size=ticks_size,
                                path_to_save=images_folder + "posterior3.pdf",
                                show_density_values=False)
    if learn_alphas:
        ticks_size = 15
        improve_jrnl_posterior_plot(jrnl, param_list[-3:], param_titles_list[-3:], label_size=label_size,
                                    title_size=title_size, path_to_save=images_folder + "posterior4.pdf",
                                    show_density_values=False, ticks_size=ticks_size)
if plot_posteriors_rhos:
    for rho_index in range(5):
        improve_jrnl_posterior_plot(jrnl, [param_list[6 + rho_index]] + [param_list[11 + rho_index]],
                                    [param_titles_list[6 + rho_index]] + [param_titles_list[11 + rho_index]],
                                    label_size=22, ticks_size=18, double_marginals_only=True,
                                    path_to_save=images_folder + "posterior_rho{}.pdf".format(rho_index + 1),
                                    show_density_values=False)
label_size = 60
title_size = 60
ticks_size = 30
if plot_posteriors_all:
    improve_jrnl_posterior_plot(jrnl, param_list, param_titles_list, label_size=label_size, title_size=title_size,
                                path_to_save=images_folder + "posterior.pdf",
                                show_density_values=False,
                                write_posterior_mean=False, show_posterior_mean=False, ticks_size=ticks_size,
                                true_parameter_values=[true_parameters_fake_dict[key] for key in
                                                       param_list] if fake_observation else None)

if plot_posteriors_marginals:
    fig, ax = jrnl.plot_posterior_distr(param_list, show_density_values=False, show_samples=False,
                                        single_marginals_only=True, write_posterior_mean=True, iteration=iteration,
                                        true_parameter_values=[true_parameters_fake_dict[key] for key in
                                                               param_list] if fake_observation else None)

    for i, key in enumerate(param_list):
        ax[i].axvline(marginal_mode[key], color="C4")
        ax[i].axvline(marginal_medians[key], color="C5")
    # mode obtained in that way is very off. Median is a bit better but still not the right way to get the MAP estimate
    # of the parameter.
    plt.savefig(images_folder + "posterior_marginals.pdf")

if plot_violinplots:
    bbox_inches_no_titles = Bbox(np.array([[0.3, .1], [5.5, 3.7]]))
    bbox_inches = Bbox(np.array([[0.3, .1], [5.5, 3.85]]))
    print("violin plots")
    # violinplots(params_array[6:16], param_titles_list[6:16], weights, showmeans=False)
    # plt.savefig(images_folder + "violinplot_rho.pdf")
    # plt.close()

    titles = ["0-19 years", "20-39 years", "40-59 years", "60-79 years", "80+ years"]
    # titles = param_titles_list[6:11]
    fig, ax = violinplots(params_array[6:11], titles, weights, showmeans=False)
    ax.set_title("Probability of needing hospitalisation when infected")
    plt.savefig(images_folder + "violinplot_rho.pdf", tightlayout=False, bbox_inches=bbox_inches)
    plt.close()
    fig, ax = violinplots(params_array[6:11], titles, weights, showmeans=False)
    plt.savefig(images_folder + "violinplot_rho_no_title.pdf", tightlayout=False, bbox_inches=bbox_inches_no_titles)
    plt.close()
    # titles = param_titles_list[11:16]
    fig, ax = violinplots(params_array[11:16], titles, weights, showmeans=False)
    ax.set_title("Probability of dying when diagnosed")
    plt.savefig(images_folder + "violinplot_rho_prime.pdf", tightlayout=False, bbox_inches=bbox_inches)
    plt.close()
    fig, ax = violinplots(params_array[11:16], titles, weights, showmeans=False)
    plt.savefig(images_folder + "violinplot_rho_prime_no_title.pdf", tightlayout=False,
                bbox_inches=bbox_inches_no_titles)
    plt.close()
    violinplots(params_array[[-3, -2, -1]], param_titles_list[-3:], weights,
                showmeans=False)
    plt.savefig(images_folder + "violinplot_alphas.pdf", tightlayout=True)
    plt.close()
    violinplots(params_array[1:6], param_titles_list[1:6], weights, showmeans=False)
    plt.savefig(images_folder + "violinplot_ds.pdf", tightlayout=True)
    plt.close()
    # violinplots(params_array[-4].reshape(1, -1), [param_titles_list[-4]], weights, showmeans=False)
    # plt.savefig(images_folder + "violinplot5.pdf")
    # plt.close()
    violinplots(params_array[0].reshape(1, -1), [param_titles_list[0]], weights, showmeans=False)
    plt.savefig(images_folder + "violinplot_beta.pdf", tightlayout=False)
    plt.close()

# should we split these in different parts?
# GENERATE SIMULATIONS FROM THE POSTERIOR SAMPLES FOR UNCERTAINTY BANDS PLOTS:
if not plot_trajectories:
    exit(0)

label_size = 16
title_size = 20
ticks_size = 16
legend_size = 12

pars_mode = np.array([marginal_mode[key] for key in marginal_mode.keys()]).squeeze()
mode_trajectory = model.forward_simulate(pars_mode, 1)[0]

pars_median = np.array([marginal_medians[key] for key in marginal_medians.keys()]).squeeze()
median_trajectory = model.forward_simulate(pars_median, 1)[0]

pars_posterior_mean = np.array([jrnl.posterior_mean()[key] for key in jrnl.posterior_mean().keys()]).squeeze()
posterior_mean_trajectory = model.forward_simulate(pars_posterior_mean, 1)[0]

# bootstrap samples
np.random.seed(seed)
post_samples = np.random.choice(range(len(weights)), p=weights.reshape(-1), size=n_post_samples)

# generate trajectories:
post_simulations_R = np.zeros((len(post_samples), model.T))
if return_observation_only_hospitalized:
    post_simulations_in_exp = np.zeros((n_post_samples, model.T + 1, observation_england_more_recent.shape[1]))
else:
    post_simulations_R_with_actual_susceptible = np.zeros((len(post_samples), model.T))
    post_simulations_in_exp = np.zeros((n_post_samples, 8, model.T + 1, model.n_age_groups))
for j, i in enumerate(post_samples):
    pars_final = [
        np.array(params[key])[i].reshape(-1) if key in params.keys() else np.array(model_variables_dict[key]).reshape(
            -1) for key in model_keys]
    post_simulations_in_exp[j] = model.forward_simulate(pars_final, 1)[0]
    post_simulations_R[j] = model.compute_R_evolution_diff_API(pars_final)
    if not (return_observation_only_hospitalized):
        # then we can compute R_t:
        post_simulations_R_with_actual_susceptible[j] = model.compute_R_evolution_with_actual_susceptible_diff_API(
            pars_final, post_simulations_in_exp[j, 0])

print(post_simulations_in_exp[0].shape, observation_england_more_recent.shape)

# generate elements for legend and axis ticks:
dates = pd.date_range(start="03-01-2020", periods=T).strftime("%d-%m")

xticks = np.arange(2, T, 14)
xticks_labels = dates[xticks]
legend_elements = [Line2D([0], [0], color='C2', lw=2, label='True data'),
                   Line2D([0], [0], color='C3', lw=2, label='Median prediction'),
                   # Line2D([0], [0], color='C6', lw=2, label='Mean param'),
                   # Line2D([0], [0], color='black', ls="dotted", label='Observation horizon')]
                   Line2D([0], [0], color='black', ls="dotted", label=r'$T_{obs}$')]
# Line2D([0], [0], color='C4', lw=2, label='Mode prediction')
# Patch(facecolor='C1', alpha=0.4, edgecolor='r', label='50% CI'),
# Patch(facecolor='C1', alpha=0.2, edgecolor='r', label='95% CI')]

legend_kwargs = {"bbox_to_anchor": (0., -.1, 1., -.1), "ncol": 3, "mode": "expand", "borderaxespad": -0.5,
                 "prop": {"size": legend_size}}
legend_kwargs_2_columns = {"bbox_to_anchor": (.1, -.1, .8, -.1), "ncol": 2, "mode": "expand", "borderaxespad": -0.5,
                           "prop": {"size": legend_size}}
bbox_inches = Bbox(np.array([[0.1, -.4], [5.6, 3.9]]))
savefig_kwargs = {"dpi": 150, "tightlayout": False, "bbox_inches": bbox_inches}

# plot of observed hospitalized people:
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
ax.axvline(41, color="black", ls=":")
ax.axvline(56, color="black", ls=":")
ax.axvline(71, color="black", ls=":")
ax.axvline(83, color="black", ls=":")

color = "tab:red"
ax.set_xticks(xticks)
ax.set_xticks(np.arange(end_step), minor=True)
ax.set_xticklabels(xticks_labels)
ax.set_xlim(start_step - 1, end_step + 1)
ax.set_xlabel("Date")
# ax.set_title("Hospitalized people in England")
ax.set_ylabel("Hospitalized people", color=color)

ax.tick_params(axis='y', labelcolor=color)
ax.plot(np.arange(start_step, end_step_observation),
        observation_england_more_recent[start_step:end_step_observation, -1], color=color, alpha=.7)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Daily deceased', color=color)  # we already handled the x-label with ax1
ax2.text(42, 130, '11th April', horizontalalignment='left', verticalalignment='center',
         rotation=90)  # , transform=ax.transAxes)
ax2.text(57, 130, '26th April', horizontalalignment='left', verticalalignment='center',
         rotation=90)  # , transform=ax.transAxes)
ax2.text(72, 750, '11th May', horizontalalignment='left', verticalalignment='center',
         rotation=90)  # , transform=ax.transAxes)
ax2.text(84, 750, '23rd May', horizontalalignment='left', verticalalignment='center',
         rotation=90)  # , transform=ax.transAxes)
# ax2.set_ylim([0, 1])
# ax2.legend()
ax2.tick_params(axis='y', labelcolor=color)
obs_deceased_sum = np.sum(observation_england_more_recent[start_step:end_step_observation, 0:5], axis=1)
# ax2.plot(np.arange(start_step, end_step_observation), obs_deceased_sum[start_step:end_step_observation], color=color,
#          alpha=.7)
ax2.plot(np.arange(start_step, end_step_observation), obs_deceased_sum, color=color,
         alpha=.7)

# ax.legend()
plt.savefig(images_folder + "observation_with_dates.pdf", dpi=savefig_kwargs["dpi"],
            tightlayout=True)
plt.close()
# exit()
# plot 1
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
ax.axvline(T_training - 1, color="black", ls=":")
ax.set_xticks(xticks)
ax.set_xticks(np.arange(end_step), minor=True)
ax.set_xticklabels(xticks_labels)
ax.set_xlim(start_step - 1, end_step + 1)
ax.set_xlabel("Date of death", size=label_size)
ax.legend(handles=legend_elements, **legend_kwargs)

if return_observation_only_hospitalized:
    obs_deceased_sum = np.sum(observation_england_more_recent[start_step:end_step_observation, 0:5], axis=1)
    ax.plot(np.arange(start_step, end_step_observation), obs_deceased_sum, color="C2")

    plot_confidence_bands(np.sum(post_simulations_in_exp[:, :, 0:5], axis=2), ax=ax, fig=fig, end_step=end_step,
                          start_step=start_step, outer_band=0, inner_band=CI_size)
    if plot_posterior_mean_traj:
        ax.plot(np.arange(start_step, end_step),
                np.sum(posterior_mean_trajectory[:, 0:5], axis=1)[start_step:end_step], color="C6")

    ax.set_title("Daily Deceased", size=title_size)
    plt.savefig(images_folder + "daily_deceased_total.pdf", **savefig_kwargs)
    plt.close()
else:
    post_simulations_total_in_exp = np.sum(post_simulations_in_exp, axis=3)

    obs_deaths_cum_total = np.cumsum(np.sum(observation_england_more_recent[:, 0:5], axis=1))

    ax.plot(np.arange(start_step, end_step_observation), obs_deaths_cum_total[start_step:end_step_observation],
            color="C2")

    plot_confidence_bands(post_simulations_total_in_exp[:, -2, :], ax=ax, fig=fig, end_step=end_step,
                          start_step=start_step, outer_band=0, inner_band=CI_size)
    ax.set_title("Cumulative Deceased", size=title_size)

    plt.savefig(images_folder + "cumulative_deceased_total.pdf", **savefig_kwargs)
    plt.close()
# plot 2
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
ax.axvline(T_training - 1, color="black", ls=":")

ax.set_xticks(xticks)
ax.set_xticks(np.arange(end_step), minor=True)
ax.set_xticklabels(xticks_labels)
ax.set_xlim(start_step - 1, end_step + 1)
ax.set_xlabel("Date of specimen sample", size=label_size)

ax.legend(handles=legend_elements, **legend_kwargs)

if return_observation_only_hospitalized:
    ax.plot(np.arange(start_step, end_step_observation),
            observation_england_more_recent[start_step:end_step_observation, -1],
            color="C2")

    plot_confidence_bands(post_simulations_in_exp[:, :, -1], ax=ax, fig=fig, end_step=end_step,
                          start_step=start_step, outer_band=0, inner_band=CI_size)
    if plot_posterior_mean_traj:
        ax.plot(np.arange(start_step, end_step), posterior_mean_trajectory[start_step:end_step, -1],
                color="C6")

    ax.set_title(r"$I^C$", size=title_size)
    ax.set_xlabel("Date", size=label_size)
    plt.savefig(images_folder + "daily_Ic_total.pdf", **savefig_kwargs)
    plt.close()
# age groups plots
for age_group in range(5):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
    ax.axvline(T_training - 1, color="black", ls=":")
    ax.legend(handles=legend_elements, **legend_kwargs)
    ax.set_xlabel("Date of death", size=label_size)
    ax.set_xticks(xticks)
    ax.set_xticks(np.arange(end_step), minor=True)
    ax.set_xticklabels(xticks_labels)
    ax.set_xlim(start_step - 1, end_step + 1)

    obs_deaths_cumsum = np.cumsum(observation_england_more_recent[start_step:end_step_observation, 0:5], axis=0)

    if return_observation_only_hospitalized:
        ax.plot(np.arange(start_step, end_step_observation),
                observation_england_more_recent[start_step:end_step_observation, age_group], color="C2")
        plot_confidence_bands(post_simulations_in_exp[:, :, age_group], ax=ax, fig=fig, end_step=end_step,
                              start_step=start_step, outer_band=0, inner_band=CI_size)
        if plot_posterior_mean_traj:
            ax.plot(np.arange(start_step, end_step),
                    posterior_mean_trajectory[start_step:end_step, age_group], color="C6")
        ax.set_title("Age group {} daily deaths".format(age_group + 1), size=title_size)
        plt.savefig(images_folder + "daily_deaths_age_group_{}.pdf".format(age_group + 1), **savefig_kwargs)

    else:
        ax.plot(np.arange(start_step, end_step_observation),
                obs_deaths_cumsum[start_step:end_step_observation, age_group], color="C2")
        plot_confidence_bands(post_simulations_in_exp[:, -2, :, age_group], ax=ax, fig=fig,
                              end_step=end_step, start_step=start_step, outer_band=0, inner_band=CI_size)
        ax.set_title("Age group {} cumulative deaths".format(age_group + 1), size=title_size)

        plt.savefig(images_folder + "cumulative_deaths_age_group_{}.pdf".format(age_group + 1), **savefig_kwargs)
    plt.close()

if not return_observation_only_hospitalized:
    # plot 3
    post_simulations_total_in_exp = np.sum(post_simulations_in_exp, axis=3)
    fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(24, 8))
    for simulation in post_simulations_total_in_exp:
        plot_results_model([np.expand_dims(simulation, -1)], show=False, fig=fig, ax=ax, color="C0",
                           alpha=.2,
                           end_step=end_step, start_step=start_step)
    ax[0, 2].plot(observation_england_more_recent[:, -1], color="C1", lw=2)
    ax[1, 2].plot(np.cumsum(np.sum(observation_england_more_recent[start_step:end_step_observation, 0:5], axis=1)), color="C1", lw=2)

    for x in range(ax.shape[0]):
        for y in range(ax.shape[1]):
            ax[x, y].axvline(T_training, color="black", ls=":")
            ax[x, y].set_xticks(xticks)
            ax[x, y].set_xticks(np.arange(end_step), minor=True)
            ax[x, y].set_xticklabels(xticks_labels)
            ax[x, y].set_xlim(start_step - 1, end_step + 1)
    plt.savefig(images_folder + "all_trajectories_sum_age_groups.pdf")
    plt.close()

    # plot 4
    fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(24, 8))
    for age_group in range(5):
        for x in range(ax.shape[0]):
            for y in range(ax.shape[1]):
                index = y + 4 * x
                plot_confidence_bands(post_simulations_in_exp[:, index, :, age_group], ax=ax[x, y], fig=fig,
                                      end_step=end_step, start_step=start_step, outer_band=0, inner_band=0,
                                      color_median="C{}".format(age_group + 1),
                                      label="Age group {}".format(age_group + 1))

    for x in range(ax.shape[0]):
        for y in range(ax.shape[1]):
            ax[x, y].axvline(T_training, color="black", ls=":")
            ax[x, y].set_xticks(xticks)
            ax[x, y].set_xticks(np.arange(end_step), minor=True)
            ax[x, y].set_xticklabels(xticks_labels)
            ax[x, y].set_xlim(start_step - 1, end_step + 1)
            ax[x, y].legend()
    ax[0, 0].set_title("Susceptible", size=title_size)
    ax[0, 1].set_title("Exposed", size=title_size)
    ax[0, 2].set_title("Infected clinical", size=title_size)
    ax[0, 3].set_title("Infected subclinical 1", size=title_size)
    ax[1, 0].set_title("Infected subclinical 2", size=title_size)
    ax[1, 1].set_title("Removed", size=title_size)
    ax[1, 2].set_title("Deceased", size=title_size)
    ax[1, 3].set_title("Confirmed cumulative", size=title_size)
    plt.savefig(images_folder + "all_trajectories_diff_age_groups.pdf")
    plt.close()

    # age groups plots for the other compartments:
    # note that the deceased one is already produced above, but here is without the true trajectory
    compartment_file_names = ["susceptible", "exposed", "infected_c", "infected_sc1", "infected_sc2", "removed",
                              "deceased", "confirmed"]
    # compartment_titles = ["Susceptible", "Exposed", "Infected Clinical", "Infected Subclinical 1",
    #                       "Infected Subclinical 2", "Removed", "Deceased", "Cumulative confirmed"]
    compartment_titles = ["$S$", "$E$", "$I^C$", "$I^{SC1}$", "$I^{SC2}$", "$R$", "$D$", "Cumulative confirmed"]
    compartment_indices = [0, 1, 2, 3, 4, 5, 6, 7]

    figsize_multiplier = 0.75
    bbox_inches = Bbox(np.array(
        [[-0.2 * figsize_multiplier, -.65 * figsize_multiplier], [5.6 * figsize_multiplier, 4 * figsize_multiplier]]))
    savefig_kwargs_2 = {"dpi": 150, "tightlayout": False, "bbox_inches": bbox_inches}
    # legend_kwargs_2 = {"bbox_to_anchor": (0.2, -.13, .8, -.13), "ncol": 2, "mode": "expand", "borderaxespad": -0.5}
    legend_kwargs_2 = {"bbox_to_anchor": (0.1, -.15, .8, -.1), "ncol": 2, "mode": "expand", "borderaxespad": -0.5,
                       "prop": {"size": legend_size}}

    for i in range(len(compartment_indices)):

        for age_group in range(5):

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6 * figsize_multiplier, 4 * figsize_multiplier))
            ax.axvline(T_training - 1, color="black", ls=":")
            ax.legend(handles=legend_elements[1:3], **legend_kwargs_2)
            ax.set_xlabel("Date", size=label_size)
            ax.set_xticks(xticks)
            ax.set_xticks(np.arange(end_step), minor=True)
            ax.set_xticklabels(xticks_labels)
            ax.set_xlim(start_step - 1, end_step + 1)

            # ax.plot(np.arange(start_step, end_step_observation),
            #         observation_england_more_recent[i, start_step:end_step_observation, age_group], color="C2")
            plot_confidence_bands(post_simulations_in_exp[:, compartment_indices[i], :, age_group], ax=ax, fig=fig,
                                  end_step=end_step, start_step=start_step, outer_band=0, inner_band=CI_size)
            if plot_posterior_mean_traj:
                ax.plot(np.arange(start_step, end_step),
                        posterior_mean_trajectory[compartment_indices[i], start_step:end_step, age_group], color="C6")
            ax.set_title(r"Age group {} {}".format(age_group + 1, compartment_titles[i]), size=title_size)

            plt.savefig(images_folder + "{}_age_group_{}.pdf".format(compartment_file_names[i], age_group + 1),
                        **savefig_kwargs_2)
            plt.close()

    # plot 5:
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
    ax.axvline(T_training - 1, color="black", ls=":")
    ax.axhline(1, color="black", ls="-", lw=1)
    # legend_elements_R = [Line2D([0], [0], color='b', lw=2, label='Median prediction')] + [legend_elements[2]]
    # ax.legend(handles=legend_elements_R, loc="upper center")  # , **legend_kwargs)
    ax.legend(handles=legend_elements[1:], **legend_kwargs_2_columns)
    ax.set_xlabel("Date", size=label_size)
    ax.set_title(r"$\mathcal{R}(t)$", size=title_size)
    ax.set_xticks(xticks)
    ax.set_xticks(np.arange(end_step), minor=True)
    ax.set_xticklabels(xticks_labels)
    ax.set_xlim(start_step - 1, end_step + 1)
    plot_confidence_bands(post_simulations_R_with_actual_susceptible, ax=ax, fig=fig, end_step=end_step,
                          start_step=start_step, outer_band=0,
                          inner_band=CI_size)
    # plt.savefig(images_folder + "R_t.pdf", dpi=savefig_kwargs["dpi"], tightlayout=savefig_kwargs["tightlayout"])
    plt.savefig(images_folder + "R_t.pdf", **savefig_kwargs)
    plt.close()

# plot 5bis:
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
ax.axvline(T_training - 1, color="black", ls=":")
ax.axhline(1, color="black", ls="-", lw=1)
# legend_elements_R = [Line2D([0], [0], color='b', lw=2, label='Median prediction')] + [legend_elements[2]]
# ax.legend(handles=legend_elements_R, loc="upper center")  # , **legend_kwargs)
ax.legend(handles=legend_elements[1:], **legend_kwargs_2_columns)
ax.set_xlabel("Date", size=label_size)
ax.set_title(r"$\mathcal{R}(t)$", size=title_size)
ax.set_xticks(xticks)
ax.set_xticks(np.arange(end_step), minor=True)
ax.set_xticklabels(xticks_labels)
ax.set_xlim(start_step - 1, end_step + 1)
plot_confidence_bands(post_simulations_R, ax=ax, fig=fig, end_step=end_step, start_step=start_step,
                      outer_band=0, inner_band=CI_size)
# plt.savefig(images_folder + "R_t.pdf", dpi=savefig_kwargs["dpi"], tightlayout=savefig_kwargs["tightlayout"])
plt.savefig(images_folder + "R_t_all_susc.pdf", **savefig_kwargs)
plt.close()
