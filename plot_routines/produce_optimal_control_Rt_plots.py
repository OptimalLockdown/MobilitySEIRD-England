import matplotlib.pyplot as plt
from abcpy.output import Journal
from matplotlib.transforms import Bbox

from optimal_control_posterior_mean import PosteriorCost
from src.distance import *
from src.models import SEI4RD
from src.utils import plot_confidence_bands

# script settings:
plot_unique = False
plot_split = True
# date_to_use = '11/04'
# date_to_use = '26/04'
date_to_use = '11/05'
# date_to_use = '23/05'
save_R_file = True
CI_size = 99
n_post_samples = 50

print("End train date:", date_to_use)
data_folder = "data/england_inference_data_1Mar_to_23May/"
if date_to_use == '23/05':
    images_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_23May/"
    results_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_23May/"
    optimal_control_results_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_23May/optimal_control_results/"
    jrnl = Journal.fromFile(results_folder + "PMCABC_inf3.jrl")
elif date_to_use == '11/05':
    images_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_11May/"
    results_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_11May/"
    optimal_control_results_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_11May/optimal_control_results/"
    jrnl = Journal.fromFile(results_folder + "journal_3.jrl")
elif date_to_use == '26/04':
    journal_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_26Apr/"
    images_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_26Apr/"
    optimal_control_results_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_26Apr/optimal_control_results/"
    jrnl = Journal.fromFile(journal_folder + "journal_3.jrl")
elif date_to_use == '11/04':
    images_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_11Apr/"
    results_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_11Apr/"
    optimal_control_results_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_11Apr/optimal_control_results/"
    jrnl = Journal.fromFile(results_folder + "journal_3_1.jrl")

alpha_home = 1  # set this to 1
# alpha_home = np.load(data_folder + "mobility_home.npy")
mobility_work = np.load(data_folder + "mobility_work.npy")
mobility_other = np.load(data_folder + "mobility_other.npy")
mobility_school = np.load(data_folder + "mobility_school.npy")

mobility_work_after_23 = np.load(data_folder + "mobility_work_after_23.npy")
mobility_other_after_23 = np.load(data_folder + "mobility_other_after_23.npy")
mobility_school_after_23 = np.load(data_folder + "mobility_school_after_23.npy")

england_pop = np.load(data_folder + "england_pop.npy", allow_pickle=True)

contact_matrix_home_england = np.load(data_folder + "contact_matrix_home_england.npy")
contact_matrix_work_england = np.load(data_folder + "contact_matrix_work_england.npy")
contact_matrix_school_england = np.load(data_folder + "contact_matrix_school_england.npy")
contact_matrix_other_england = np.load(data_folder + "contact_matrix_other_england.npy")

# get training data
dt = 0.1  # integration timestep
T = mobility_school.shape[0] - 1  # horizon time in days (needs to be 1 less than the number of days in observation)
total_population = england_pop  # population for each age group
# 16th March: Boris Johnson asked old people to isolate; we then learn a new alpha from the 18th March:
lockdown_day = 17

# alpha_home = np.repeat(alpha_home, np.int(1 / dt), axis=0)
mobility_work = np.repeat(mobility_work, np.int(1 / dt), axis=0)
mobility_other = np.repeat(mobility_other, np.int(1 / dt), axis=0)
mobility_school = np.repeat(mobility_school, np.int(1 / dt), axis=0)

# ABC model (priors need to be fixed better):

# we initialize the model with fixed values, it should not be a problem.
model = SEI4RD([0.5] * 20, tot_population=total_population, T=T,
               contact_matrix_school=contact_matrix_school_england, contact_matrix_work=contact_matrix_work_england,
               contact_matrix_home=contact_matrix_home_england, contact_matrix_other=contact_matrix_other_england,
               alpha_school=mobility_school, alpha_work=mobility_work, alpha_home=alpha_home,
               alpha_other=mobility_other, modify_alpha_home=False, dt=dt, return_once_a_day=True,
               learn_alphas_old=True, lockdown_day=lockdown_day)

# extract posterior sample points and bootstrap them:
seed = 1
np.random.seed(seed)
iteration = - 1
weights = jrnl.get_weights(iteration) / np.sum(jrnl.get_weights(iteration))
params = jrnl.get_parameters(iteration)
# bootstrap
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

# instantiate the posterior cost class:
posterior_cost = PosteriorCost(model, phi_func_sc=lambda x: x, phi_func_death=lambda x: x, beta_vals=beta_values,
                               kappa_vals=kappa_values, gamma_c_vals=gamma_c_values, gamma_r_vals=gamma_r_values,
                               gamma_rc_vals=gamma_rc_values, nu_vals=nu_values, rho_vals=rho_values,
                               rho_prime_vals=rho_prime_values, alpha_123_vals=alpha_123_values,
                               alpha_4_vals=alpha_4_values, alpha_5_vals=alpha_5_values,
                               initial_exposed_vals=initial_exposed_values)

# get the R values and plot them for the different files:

if date_to_use == '23/05':
    files_list = [
        "mode_eps_50_100_1_window_30_post_samples_50mobility_iterative_63",
        "mode_eps_100_200_2_window_30_post_samples_50mobility_iterative_63",
        "mode_eps_150_300_3_window_30_post_samples_50mobility_iterative_63",
        "mode_eps_10_100_1_window_30_post_samples_50mobility_iterative_48",
        "mode_eps_20_200_2_window_30_post_samples_50mobility_iterative_48",
        "mode_eps_30_300_3_window_30_post_samples_50mobility_iterative_49",
        # "mode_eps_100_100_100_window_30mobility_iterative",
        # "mode_eps_200_200_200_window_30mobility_iterative",
        # "mode_eps_300_300_300_window_30mobility_iterative"
    ]  # , "mode_eps_500_500_500_window_30mobility_iterative"]
    labels_list = [
        [50, 100, 1],
        [100, 200, 2],
        [150, 300, 3],
        [10, 100, 1],
        [20, 200, 2],
        [30, 300, 3],
        # [100, 100, 100],
        # [200, 200, 200],
        # [300, 300, 300]
    ]  # , [500, 500, 500]]
elif date_to_use == '11/05':
    files_list = [
        "mode_eps_50_100_1_window_30_post_samples_50mobility_iterative_101",
        "mode_eps_100_200_2_window_30_post_samples_50mobility_iterative_65",
        "mode_eps_150_300_3_window_30_post_samples_50mobility_iterative_67",
        "mode_eps_10_100_1_window_30_post_samples_50mobility_iterative_52",
        "mode_eps_20_200_2_window_30_post_samples_50mobility_iterative_51",
        "mode_eps_30_300_3_window_30_post_samples_50mobility_iterative_51",
        # "mode_eps_100_100_100_window_30mobility_iterative",
        # "mode_eps_200_200_200_window_30mobility_iterative",
        # "mode_eps_300_300_300_window_30mobility_iterative"
    ]  # , "mode_eps_500_500_500_window_30mobility_iterative"]
    labels_list = [
        [50, 100, 1],
        [100, 200, 2],
        [150, 300, 3],
        [10, 100, 1],
        [20, 200, 2],
        [30, 300, 3],
        # [100, 100, 100],
        # [200, 200, 200],
        # [300, 300, 300]
    ]  # , [500, 500, 500]]
elif date_to_use == '26/04':
    files_list = [
        "mode_eps_150_300_3_window_30_post_samples_50mobility_iterative_28",
    ]
    labels_list = [
        [150, 300, 3],
    ]
elif date_to_use == '11/04':
    files_list = [
        "mode_eps_150_300_3_window_30_post_samples_50mobility_iterative_81",
    ]
    labels_list = [
        [150, 300, 3],
    ]

colors_list = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
# ticks for 23 May:
if date_to_use == '23/05':
    n_days = 48
    ticks = [0, 9, 19, 29, 39, 47]
    ticks_labels = ['23/05', '02/06', '12/06', '22/06', '02/07', '11/07']
elif date_to_use == '11/05':
    n_days = 51
    ticks = [0, 9, 19, 29, 39, 49]
    ticks_labels = ['11/05', '20/05', '30/05', '10/06', '20/06', '30/06']
elif date_to_use == '26/04':
    n_days = 28
    # ticks = [0, 9, 19, 29, 39, 49, 59, 69, 79]
    # ticks_labels = ['26/04', '20/04', '30/04', '10/05', '20/05', '30/05', '10/06', '20/06', '30/06']
elif date_to_use == '11/04':
    n_days = 81
    ticks = [0, 9, 19, 29, 39, 49, 59, 69, 79]
    ticks_labels = ['11/04', '20/04', '30/04', '10/05', '20/05', '30/05', '10/06', '20/06', '30/06']

# manipolate the measured mobility values from Google data after end of training period: we assume that they stay
# constant for the prediction horizon:
mobility_school_after_23 = np.concatenate(
    (mobility_school_after_23, [mobility_school_after_23[-1]] * (n_days - len(mobility_school_after_23))))
mobility_work_after_23 = np.concatenate(
    (mobility_work_after_23, [mobility_work_after_23[-1]] * (n_days - len(mobility_work_after_23))))
mobility_other_after_23 = np.concatenate(
    (mobility_other_after_23, [mobility_other_after_23[-1]] * (n_days - len(mobility_other_after_23))))
mobility_after_23 = np.stack((mobility_school_after_23, mobility_work_after_23, mobility_other_after_23))
np.save(optimal_control_results_folder + "measured_mobility_values_after23May.npy", mobility_after_23)
files_list.append("measured_mobility_values_after23May")
mobility_school_after_11 = np.concatenate((mobility_school[-120:][::10], mobility_school_after_23[:-12]))
mobility_work_after_11 = np.concatenate((mobility_work[-120:][::10], mobility_work_after_23[:-12]))
mobility_other_after_11 = np.concatenate((mobility_other[-120:][::10], mobility_other_after_23[:-12]))
mobility_after_11 = np.stack((mobility_school_after_11, mobility_work_after_11, mobility_other_after_11))
np.save(optimal_control_results_folder + "measured_mobility_values_after11May.npy", mobility_after_11)
files_list.append("measured_mobility_values_after11May")
if date_to_use == '26/04':
    pass
elif date_to_use == '11/04':
    pass
    # files_list.append("measured_mobility_values_after11Apr")

labels_list.append("Measured mobility")
colors_list.append("C9")
if plot_unique:
    # PLOT WITH ALL THINGS TOGETHER:
    fig, ax = plt.subplots(figsize=(8, 4))
    # fig2, ax2 = plt.subplots(figsize=(8, 4))
    # fig3, ax3 = plt.subplots(figsize=(8, 4))

    for i, file in enumerate(files_list):
        mobilities = np.load(optimal_control_results_folder + file + ".npy")
        # print(mobilities.shape)
        mobilities = mobilities[:, 0:n_days]

        R_optimal_control = posterior_cost.evolve_and_compute_R(n_days, mobilities)

        if save_R_file:
            np.save(optimal_control_results_folder + file + "_R_values.npy", R_optimal_control)

        # confidence bands do not look very good, they overlap quite a lot.
        # plot_confidence_bands(R_optimal_control, ax=ax, fig=fig, outer_band=0, inner_band=CI_size,
        #                       label=r"$\epsilon = [{},{},{}]$".format(*labels_list[i]) if isinstance(labels_list[i],
        #                                                                                              list) else
        #                       labels_list[i], color_median=colors_list[i], color_inner=colors_list[i],
        #                       ls="-" if isinstance(labels_list[i], list) else "--", alpha_median=.7, alpha_inner=0,
        #                       fill_between=True)
        ax.plot(np.median(R_optimal_control, axis=0),
                label=r"$\epsilon = [{},{},{}]$".format(*labels_list[i]) if isinstance(labels_list[i], list) else
                labels_list[i], ls="-" if isinstance(labels_list[i], list) else "--", color=colors_list[i], alpha=0.7)

        # ax2.plot(np.sum(infected_c_evolution[::10], axis=1),
        #          label=r"$\epsilon = [{},{},{}]$".format(*labels_list[i]) if isinstance(labels_list[i], list) else
        #          labels_list[i], ls="-" if isinstance(labels_list[i], list) else "--", color=colors_list[i])
        # ax3.plot(np.sum(deceased_evolution[::10] - deceased_evolution[0], axis=1),
        #          label=r"$\epsilon = [{},{},{}]$".format(*labels_list[i]) if isinstance(labels_list[i], list) else
        #          labels_list[i], ls="-" if isinstance(labels_list[i], list) else "--", color=colors_list[i])
        # print(np.sum(deceased_evolution[-1] - deceased_evolution[0]))

        # print(R_all_susc_optimal_control)
    # set up ax
    ax.set_ylabel(r"$\mathcal{R}(t)$")
    ax.set_xlabel("Date")
    ax.set_xticks(ticks)
    ax.set_xticks(np.arange(n_days), minor=True)
    ax.set_xticklabels(ticks_labels)

    ax.axhline(1, color="black", ls="dotted", lw=1)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    legend_x = 1
    legend_y = 0.5
    legend_kwargs = {"bbox_to_anchor": (legend_x, legend_y), "ncol": 1,
                     # "mode": "expand", "borderaxespad": -0.5,
                     "loc": "center left"}

    ax.legend(**legend_kwargs)
    bbox_inches = Bbox(np.array([[0.2, .2], [7.6, 3.8]]))
    savefig_kwargs = {"dpi": 150, "tightlayout": False, "bbox_inches": bbox_inches}

    if date_to_use == '23/05':
        pass
        # ax.set_title(r"Evolution of $\mathcal{R}(t)$ (training until 23rd May)")
    elif date_to_use == '11/05':
        pass
        # ax.set_title(r"Evolution of $\mathcal{R}(t)$ (training until 11th May)")
    elif date_to_use == '11/04':
        ax.set_title(r"Evolution of $\mathcal{R}(t)$ (training until 11th Apr)")
    fig.savefig(optimal_control_results_folder + "R_all_susc_evolution.pdf", **savefig_kwargs)

    # set up ax2
    # ax2.set_ylabel(r"$I^C(t)$")
    # ax2.set_xlabel("Date")
    # ax2.set_xticks(ticks)
    # ax2.set_xticklabels(ticks_labels)
    #
    # # ax2.axhline(1, color="black", ls="dotted", lw=1)
    # box = ax2.get_position()
    # ax2.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # legend_x = 1
    # legend_y = 0.5
    # legend_kwargs = {"bbox_to_anchor": (legend_x, legend_y), "ncol": 1,
    #                  # "mode": "expand", "borderaxespad": -0.5,
    #                  "loc": "center left"}
    #
    # ax2.legend(**legend_kwargs)
    # bbox_inches = Bbox(np.array([[0.2, .2], [7.6, 3.8]]))
    # savefig_kwargs = {"dpi": 150, "tightlayout": False, "bbox_inches": bbox_inches}
    #
    # # ax2.set_title(r"Evolution of $\mathcal{R}(t)$")
    # # plt.savefig(fig2, optimal_control_results_folder + "R_evolution.pdf", **savefig_kwargs)
    # ax2.set_title(r"Evolution of $I^C(t)$")
    # fig2.savefig(optimal_control_results_folder + "I_C_evolution.pdf", **savefig_kwargs)
    # fig2.show()
    #
    # # set up ax3
    # ax3.set_ylabel(r"$D(t)$")
    # ax3.set_xlabel("Date")
    # ax3.set_xticks(ticks)
    # ax3.set_xticklabels(ticks_labels)
    #
    # # ax3.axhline(1, color="black", ls="dotted", lw=1)
    # box = ax3.get_position()
    # ax3.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # legend_x = 1
    # legend_y = 0.5
    # legend_kwargs = {"bbox_to_anchor": (legend_x, legend_y), "ncol": 1,
    #                  # "mode": "expand", "borderaxespad": -0.5,
    #                  "loc": "center left"}
    #
    # ax3.legend(**legend_kwargs)
    # bbox_inches = Bbox(np.array([[0.2, .2], [7.6, 3.8]]))
    # savefig_kwargs = {"dpi": 150, "tightlayout": False, "bbox_inches": bbox_inches}
    #
    # # ax3.set_title(r"Evolution of $\mathcal{R}(t)$")
    # # plt.savefig(fig3, optimal_control_results_folder + "R_evolution.pdf", **savefig_kwargs)
    # ax3.set_title(r"Evolution of $D(t)$")
    # fig3.savefig(optimal_control_results_folder + "D_evolution.pdf", **savefig_kwargs)
    # fig3.show()

# SEPARATE PLOTS:
# we create a list of lists in order to create different plots:

if plot_split:
    # files_list_1 = [files_list[-1]] + files_list[0:3]
    # files_list_2 = [files_list[-1]] + files_list[3:6]
    # files_list_3 = [files_list[-1]] + files_list[6:9]
    # labels_list_1 = [labels_list[-1]] + labels_list[0:3]
    # labels_list_2 = [labels_list[-1]] + labels_list[3:6]
    # labels_list_3 = [labels_list[-1]] + labels_list[6:9]
    # colors_list_1 = [colors_list[-1]] + colors_list[0:3]
    # colors_list_2 = [colors_list[-1]] + colors_list[3:6]
    # colors_list_3 = [colors_list[-1]] + colors_list[6:9]
    #
    # # colors_list = []
    # list_of_lists = zip([files_list_1, files_list_2, files_list_3], [labels_list_1, labels_list_2, labels_list_3],
    #                     [colors_list_1, colors_list_2, colors_list_3])

    files_splitted_lists = [[files_list[-1]] + [files_list[i]] for i in range(len(files_list) - 1)]
    labels_splitted_lists = [[labels_list[-1]] + [labels_list[i]] for i in range(len(labels_list) - 1)]
    colors_splitted_lists = [[colors_list[-1]] + [colors_list[i]] for i in range(len(colors_list) - 1)]

    list_of_lists = zip(files_splitted_lists, labels_splitted_lists, colors_splitted_lists)
    figsize_multiplier = 0.75
    for list_id, lists in enumerate(list_of_lists):
        files_list, labels_list, colors_list = lists
        fig, ax = plt.subplots(figsize=(6 * figsize_multiplier, 4 * figsize_multiplier))
        # fig2, ax2 = plt.subplots(figsize=(6 * figsize_multiplier, 4 * figsize_multiplier))
        # fig3, ax3 = plt.subplots(figsize=(6 * figsize_multiplier, 4 * figsize_multiplier))
        for i, file in enumerate(files_list):
            mobilities = np.load(optimal_control_results_folder + file + ".npy")
            mobilities = mobilities[:, 0:n_days]

            # add stuff here; this is not super efficient, it repeats some computation.
            R_optimal_control = posterior_cost.evolve_and_compute_R(n_days, mobilities)

            # confidence bands do not look very good, they overlap quite a lot.
            plot_confidence_bands(R_optimal_control, ax=ax, fig=fig, outer_band=0, inner_band=CI_size,
                                  label=r"$\epsilon = [{},{},{}]$".format(*labels_list[i]) if isinstance(labels_list[i],
                                                                                                         list) else
                                  labels_list[i], color_median=colors_list[i], color_inner=colors_list[i],
                                  ls="-" if isinstance(labels_list[i], list) else "--", alpha_median=.7,
                                  alpha_inner=0.35,
                                  fill_between=True)
            # ax.plot(np.median(R_optimal_control, axis=0),
            #         label=r"$\epsilon = [{},{},{}]$".format(*labels_list[i]) if isinstance(labels_list[i], list) else
            #         labels_list[i], ls="-" if isinstance(labels_list[i], list) else "--", color=colors_list[i],
            #         alpha=0.7)

            # ax2.plot(np.sum(infected_c_evolution[::10], axis=1),
            #          label=r"$\epsilon = [{},{},{}]$".format(*labels_list[i]) if isinstance(labels_list[i], list) else
            #          labels_list[i], ls="-" if isinstance(labels_list[i], list) else "--", color=colors_list[i])
            # ax3.plot(np.sum(deceased_evolution[::10] - deceased_evolution[0], axis=1),
            #          label=r"$\epsilon = [{},{},{}]$".format(*labels_list[i]) if isinstance(labels_list[i], list) else
            #          labels_list[i], ls="-" if isinstance(labels_list[i], list) else "--", color=colors_list[i])

            # print(R_all_susc_optimal_control)
        ax.set_ylabel(r"$\mathcal{R}(t)$")
        ax.set_xlabel("Date")
        ax.set_xticks(ticks)
        ax.set_xticks(np.arange(n_days), minor=True)
        ax.set_xticklabels(ticks_labels)

        ax.axhline(1, color="black", ls="dotted", lw=1)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        # legend_x = 1
        # legend_y = 0.5
        # legend_kwargs = {"bbox_to_anchor": (legend_x, legend_y), "ncol": 1,
        #                  # "mode": "expand", "borderaxespad": -0.5,
        #                  "loc": "center left"}

        legend_kwargs = {}
        ax.legend(**legend_kwargs)
        bbox_inches = Bbox(np.array(
            [[-0.15 * figsize_multiplier, -0.17 * figsize_multiplier],
             [5.6 * figsize_multiplier, 3.6 * figsize_multiplier]]))
        savefig_kwargs = {"dpi": 150, "tightlayout": True, "bbox_inches": bbox_inches}

        if date_to_use == '23/05':
            pass
            # ax.set_title(r"Evolution of $\mathcal{R}(t)$ (training until 23rd May)")
        elif date_to_use == '11/05':
            pass
            # ax.set_title(r"Evolution of $\mathcal{R}(t)$ (training until 11th May)")
        elif date_to_use == '11/04':
            ax.set_title(r"Evolution of $\mathcal{R}(t)$ (training until 11th Apr)")
        fig.savefig(optimal_control_results_folder + "R_all_susc_evolution_split_{}.pdf".format(list_id),
                    **savefig_kwargs)
        # exit()
        # fig.show()

        # # set up ax2
        # ax2.set_ylabel(r"$I^C(t)$")
        # ax2.set_xlabel("Date")
        # ax2.set_xticks(ticks)
        # ax2.set_xticklabels(ticks_labels)
        #
        # # ax2.axhline(1, color="black", ls="dotted", lw=1)
        # # box = ax2.get_position()
        # # ax2.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        # # legend_x = 1
        # # legend_y = 0.5
        # # legend_kwargs = {"bbox_to_anchor": (legend_x, legend_y), "ncol": 1,
        # #                  # "mode": "expand", "borderaxespad": -0.5,
        # #                  "loc": "center left"}
        #
        # legend_kwargs = {}
        # ax2.legend(**legend_kwargs)
        # bbox_inches = Bbox(np.array(
        #     [[0 * figsize_multiplier, 0 * figsize_multiplier], [5.8 * figsize_multiplier, 3.9 * figsize_multiplier]]))
        # savefig_kwargs = {"dpi": 150, "tightlayout": False, "bbox_inches": bbox_inches}
        #
        # # ax2.set_title(r"Evolution of $\mathcal{R}(t)$")
        # # plt.savefig(fig2, optimal_control_results_folder + "R_evolution.pdf", **savefig_kwargs)
        # ax2.set_title(r"Evolution of $I^C(t)$")
        # fig2.savefig(optimal_control_results_folder + "I_C_evolution_split_{}.pdf".format(list_id), **savefig_kwargs)
        # fig2.show()
        #
        # # set up ax3
        # ax3.set_ylabel(r"$D(t)$")
        # ax3.set_xlabel("Date")
        # ax3.set_xticks(ticks)
        # ax3.set_xticklabels(ticks_labels)
        #
        # # ax3.axhline(1, color="black", ls="dotted", lw=1)
        # # box = ax3.get_position()
        # # ax3.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        # # legend_x = 1
        # # legend_y = 0.5
        # # legend_kwargs = {"bbox_to_anchor": (legend_x, legend_y), "ncol": 1,
        # #                  # "mode": "expand", "borderaxespad": -0.5,
        # #                  "loc": "center left"}
        #
        # legend_kwargs = {}
        # ax3.legend(**legend_kwargs)
        # bbox_inches = Bbox(np.array(
        #     [[0 * figsize_multiplier, 0 * figsize_multiplier], [5.8 * figsize_multiplier, 3.9 * figsize_multiplier]]))
        # savefig_kwargs = {"dpi": 150, "tightlayout": True, "bbox_inches": bbox_inches}
        #
        # # ax3.set_title(r"Evolution of $\mathcal{R}(t)$")
        # # plt.savefig(fig3, optimal_control_results_folder + "R_evolution.pdf", **savefig_kwargs)
        # ax3.set_title(r"Evolution of $D(t)$")
        # fig3.savefig(optimal_control_results_folder + "D_evolution_split_{}.pdf".format(list_id), **savefig_kwargs)
        # fig3.show()
