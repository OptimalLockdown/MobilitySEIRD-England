import numpy as np
import pylab as plt
from matplotlib.transforms import Bbox

from src.utils import plot_confidence_bands


def plot_mobilites_MAP_expected(T, mobility_matrix_MAP, mobility_matrix_expected):
    fig, ax1 = plt.subplots(1, figsize=(6, 4))
    if T == 41:
        ax1.set_xticks([20, 51, 81, 112])
        ax1.set_xticklabels(['01/05', '01/06', '01/07', '01/08'])
    elif T == 71:
        ax1.set_xticks([19, 49, 80, 111])
        ax1.set_xticklabels(['01/06', '01/07', '01/08', '01/09'])
    elif T == 83:
        ax1.set_xticks([7, 37, 68, 99])
        ax1.set_xticklabels(['01/06', '01/07', '01/08', '01/09'])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Mobility')  # we already handled the x-label with ax1
    ax1.set_ylim([0, 1])
    color = 'tab:red'
    ax1.plot(mobility_matrix_MAP[1, :], linestyle='solid', label=r'$work_{MAP}$', color=color, alpha=0.6)
    ax1.plot(mobility_matrix_MAP[0, :], linestyle='dashed', label=r'$school_{MAP}$', color=color, alpha=0.6)
    ax1.plot(mobility_matrix_MAP[2, :], linestyle='dashdot', label=r'$other_{MAP}$', color=color, alpha=0.6)

    color = 'tab:blue'
    ax1.plot(mobility_matrix_expected[1, :], linestyle='solid', label=r'$work_{expected}$', color=color, alpha=0.6)
    ax1.plot(mobility_matrix_expected[0, :], linestyle='dashed', label=r'$school_{expected}$', color=color, alpha=0.6)
    ax1.plot(mobility_matrix_expected[2, :], linestyle='dashdot', label=r'$other_{expected}$', color=color, alpha=0.6)

    ax1.legend()

    return fig, ax1


def plot_mobilites_compare_11_23May(mobility_matrix_11, mobility_matrix_23, ticks=None, ticklabels=None):
    fig, ax1 = plt.subplots(1, figsize=(6, 4))
    if ticks is None or ticklabels is None:
        ax1.set_xticks([19, 49, 80, 111])
        ax1.set_xticklabels(['01/06', '01/07', '01/08', '01/09'])
    else:
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(ticklabels)
        ax1.set_xticks(np.arange(51), minor=True)  # bad hard-coded
    ax1.set_xlabel('Date')

    ax1.set_ylabel('Mobility')  # we already handled the x-label with ax1
    ax1.set_ylim([0, 1])
    color = 'tab:red'
    ax1.plot(mobility_matrix_11[1, :], linestyle='solid', label=r'$work$ (11th May)', color=color, alpha=0.6)
    ax1.plot(mobility_matrix_11[0, :], linestyle='dashed', label=r'$school$ (11th May)', color=color, alpha=0.6)
    ax1.plot(mobility_matrix_11[2, :], linestyle='dashdot', label=r'$other$ (11th May)', color=color, alpha=0.6)

    color = 'tab:blue'
    len_timeseries = mobility_matrix_23.shape[1]
    ax1.plot(np.arange(len_timeseries) + 12, mobility_matrix_23[1, :], linestyle='solid', label=r'$work$ (23rd May)',
             color=color, alpha=0.6)
    ax1.plot(np.arange(len_timeseries) + 12, mobility_matrix_23[0, :], linestyle='dashed', label=r'$school$ (23rd May)',
             color=color, alpha=0.6)
    ax1.plot(np.arange(len_timeseries) + 12, mobility_matrix_23[2, :], linestyle='dashdot', label=r'$other$ (23rd May)',
             color=color, alpha=0.6)

    ax1.legend(ncol=2, loc='upper center', framealpha=1)

    return fig, ax1


def plot_mobilites_compare_11Apr_11May_23May(mobility_matrix_11Apr, mobility_matrix_11May, mobility_matrix_23May,
                                             ticks=None, ticklabels=None):
    fig, ax1 = plt.subplots(1, figsize=(6, 4))
    if ticks is None or ticklabels is None:
        ax1.set_xticks([19, 49, 79, 110, 141])
        ax1.set_xticklabels(['01/05', '01/06', '01/07', '01/08', '01/09'])
    else:
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(ticklabels)
        ax1.set_xticks(np.arange(81), minor=True)  # bad hard-coded
    ax1.set_xlabel('Date')

    ax1.set_ylabel('Mobility')  # we already handled the x-label with ax1
    ax1.set_ylim([0, 1])
    color = 'tab:green'
    ax1.plot(mobility_matrix_11Apr[1, :], linestyle='solid', label=r'$work_{{11th\ Apr}}$', color=color, alpha=0.6)
    ax1.plot(mobility_matrix_11Apr[0, :], linestyle='dashed', label=r'$school_{{11th\ Apr}}$', color=color, alpha=0.6)
    ax1.plot(mobility_matrix_11Apr[2, :], linestyle='dashdot', label=r'$other_{{11th\ Apr}}$', color=color, alpha=0.6)

    color = 'tab:red'
    len_timeseries = mobility_matrix_11May.shape[1]
    ax1.plot(np.arange(len_timeseries) + 30, mobility_matrix_11May[1, :], linestyle='solid',
             label=r'$work_{{11th\ May}}$',
             color=color, alpha=0.6)
    ax1.plot(np.arange(len_timeseries) + 30, mobility_matrix_11May[0, :], linestyle='dashed',
             label=r'$school_{{11th\ May}}$', color=color, alpha=0.6)
    ax1.plot(np.arange(len_timeseries) + 30, mobility_matrix_11May[2, :], linestyle='dashdot',
             label=r'$other_{{11th\ May}}$', color=color, alpha=0.6)

    color = 'tab:blue'
    len_timeseries = mobility_matrix_23May.shape[1]
    ax1.plot(np.arange(len_timeseries) + 42, mobility_matrix_23May[1, :], linestyle='solid',
             label=r'$work_{{23rd\ May}}$', color=color, alpha=0.6)
    ax1.plot(np.arange(len_timeseries) + 42, mobility_matrix_23May[0, :], linestyle='dashed',
             label=r'$school_{{23rd\ May}}$',
             color=color, alpha=0.6)
    ax1.plot(np.arange(len_timeseries) + 42, mobility_matrix_23May[2, :], linestyle='dashdot',
             label=r'$other_{{23rd\ May}}$',
             color=color, alpha=0.6)

    ax1.legend(ncol=3, loc='upper center', framealpha=1)

    return fig, ax1


def plot_mobilites_compare_11Apr_26Apr_11May_23May(mobility_matrix_11Apr, mobility_matrix_26Apr, mobility_matrix_11May,
                                                   mobility_matrix_23May, ticks=None, ticklabels=None):
    fig, ax1 = plt.subplots(1, figsize=(8, 4))
    if ticks is None or ticklabels is None:
        ax1.set_xticks([19, 49, 79, 110, 141])
        ax1.set_xticklabels(['01/05', '01/06', '01/07', '01/08', '01/09'])
    else:
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(ticklabels)
        ax1.set_xticks(np.arange(81), minor=True)  # bad hard-coded
    ax1.set_xlabel('Date')

    ax1.set_ylabel('Mobility')  # we already handled the x-label with ax1
    ax1.set_ylim([0, 1])
    color = 'tab:red'
    ax1.plot(mobility_matrix_11Apr[1, :], linestyle='solid', label=r'$work_{{11th\ Apr}}$', color=color, alpha=0.6)
    ax1.plot(mobility_matrix_11Apr[0, :], linestyle='dashed', label=r'$school_{{11th\ Apr}}$', color=color, alpha=0.6)
    ax1.plot(mobility_matrix_11Apr[2, :], linestyle='dashdot', label=r'$other_{{11th\ Apr}}$', color=color, alpha=0.6)

    color = 'tab:blue'
    len_timeseries = mobility_matrix_26Apr.shape[1]
    ax1.plot(np.arange(len_timeseries) + 15, mobility_matrix_26Apr[1, :], linestyle='solid',
             label=r'$work_{{26th\ Apr}}$', color=color, alpha=0.6)
    ax1.plot(np.arange(len_timeseries) + 15, mobility_matrix_26Apr[0, :], linestyle='dashed',
             label=r'$school_{{26th\ Apr}}$', color=color, alpha=0.6)
    ax1.plot(np.arange(len_timeseries) + 15, mobility_matrix_26Apr[2, :], linestyle='dashdot',
             label=r'$other_{{26th\ Apr}}$', color=color, alpha=0.6)

    color = 'tab:green'
    len_timeseries = mobility_matrix_11May.shape[1]
    ax1.plot(np.arange(len_timeseries) + 30, mobility_matrix_11May[1, :], linestyle='solid',
             label=r'$work_{{11th\ May}}$',
             color=color, alpha=0.6)
    ax1.plot(np.arange(len_timeseries) + 30, mobility_matrix_11May[0, :], linestyle='dashed',
             label=r'$school_{{11th\ May}}$', color=color, alpha=0.6)
    ax1.plot(np.arange(len_timeseries) + 30, mobility_matrix_11May[2, :], linestyle='dashdot',
             label=r'$other_{{11th\ May}}$', color=color, alpha=0.6)

    color = 'tab:orange'
    len_timeseries = mobility_matrix_23May.shape[1]
    ax1.plot(np.arange(len_timeseries) + 42, mobility_matrix_23May[1, :], linestyle='solid',
             label=r'$work_{{23rd\ May}}$', color=color, alpha=0.6)
    ax1.plot(np.arange(len_timeseries) + 42, mobility_matrix_23May[0, :], linestyle='dashed',
             label=r'$school_{{23rd\ May}}$',
             color=color, alpha=0.6)
    ax1.plot(np.arange(len_timeseries) + 42, mobility_matrix_23May[2, :], linestyle='dashdot',
             label=r'$other_{{23rd\ May}}$',
             color=color, alpha=0.6)

    ax1.legend(ncol=4, loc='upper center', framealpha=1)

    return fig, ax1


def plot_R_values_compare_11Apr_11May_23May(R_values_11Apr, R_values_11May, R_values_23May,
                                            ticks=None, ticklabels=None, CI_size=99):
    fig, ax1 = plt.subplots(1, figsize=(6, 4))
    if ticks is None or ticklabels is None:
        ax1.set_xticks([19, 49, 79, 110, 141])
        ax1.set_xticklabels(['01/05', '01/06', '01/07', '01/08', '01/09'])
    else:
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(ticklabels)
        ax1.set_xticks(np.arange(81), minor=True)  # bad hard-coded
    ax1.set_xlabel('Date')

    ax1.set_ylabel(r'$\mathcal{R}(t)$')  # we already handled the x-label with ax1
    color = 'tab:green'
    plot_confidence_bands(R_values_11Apr, ax=ax1, fig=fig, outer_band=0, inner_band=CI_size, label="11th Apr",
                          color_median=color, color_inner=color, alpha_median=.7, alpha_inner=0.35, fill_between=True, )

    color = 'tab:red'
    plot_confidence_bands(np.concatenate((np.zeros((50, 30)), R_values_11May), axis=1), start_step=30,
                          ax=ax1, fig=fig, outer_band=0, inner_band=CI_size, label="11th May",
                          color_median=color, color_inner=color, alpha_median=.7, alpha_inner=0.35, fill_between=True, )

    color = 'tab:blue'
    plot_confidence_bands(np.concatenate((np.zeros((50, 42)), R_values_23May), axis=1), start_step=42,
                          ax=ax1, fig=fig, outer_band=0, inner_band=CI_size, label="23rd May",
                          color_median=color, color_inner=color, alpha_median=.7, alpha_inner=0.35, fill_between=True, )

    ax1.legend(ncol=3, loc='upper center', framealpha=1)

    return fig, ax1


def plot_R_values_compare_11Apr_26Apr_11May_23May(R_values_11Apr, R_values_26Apr, R_values_11May, R_values_23May,
                                                  ticks=None, ticklabels=None, CI_size=99):
    fig, ax1 = plt.subplots(1, figsize=(8, 4))
    if ticks is None or ticklabels is None:
        ax1.set_xticks([19, 49, 79, 110, 141])
        ax1.set_xticklabels(['01/05', '01/06', '01/07', '01/08', '01/09'])
    else:
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(ticklabels)
        ax1.set_xticks(np.arange(81), minor=True)  # bad hard-coded
    ax1.set_xlabel('Date')

    ax1.set_ylabel(r'$\mathcal{R}(t)$')  # we already handled the x-label with ax1
    color = 'tab:red'
    plot_confidence_bands(R_values_11Apr, ax=ax1, fig=fig, outer_band=0, inner_band=CI_size, label="11th Apr",
                          color_median=color, color_inner=color, alpha_median=.7, alpha_inner=0.3, fill_between=True, )

    color = 'tab:blue'
    plot_confidence_bands(np.concatenate((np.zeros((50, 15)), R_values_26Apr), axis=1), start_step=15,
                          ax=ax1, fig=fig, outer_band=0, inner_band=CI_size, label="26th Apr",
                          color_median=color, color_inner=color, alpha_median=.7, alpha_inner=0.3, fill_between=True, )
    color = 'tab:green'
    plot_confidence_bands(np.concatenate((np.zeros((50, 30)), R_values_11May), axis=1), start_step=30,
                          ax=ax1, fig=fig, outer_band=0, inner_band=CI_size, label="11th May",
                          color_median=color, color_inner=color, alpha_median=.7, alpha_inner=0.3, fill_between=True, )

    color = 'tab:orange'
    plot_confidence_bands(np.concatenate((np.zeros((50, 42)), R_values_23May), axis=1), start_step=42,
                          ax=ax1, fig=fig, outer_band=0, inner_band=CI_size, label="23rd May",
                          color_median=color, color_inner=color, alpha_median=.7, alpha_inner=0.3, fill_between=True, )

    ax1.legend(ncol=4, loc='upper center', framealpha=1)

    return fig, ax1


if __name__ == '__main__':
    optimal_control_results_folder_11Apr = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_11Apr/optimal_control_results/"
    optimal_control_results_folder_26Apr = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_26Apr/optimal_control_results/"
    optimal_control_results_folder_11May = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_11May/optimal_control_results/"
    optimal_control_results_folder_23May = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_23May/optimal_control_results/"
    T_11 = 71
    T_23 = 83
    names_files_MAP = ["mode_eps_10_100_1_window_30mobility_iterative", "mode_eps_20_200_2_window_30mobility_iterative",
                       "mode_eps_30_300_3_window_30mobility_iterative", "mode_eps_50_100_1_window_30mobility_iterative",
                       "mode_eps_100_100_100_window_30mobility_iterative",
                       "mode_eps_100_200_2_window_30mobility_iterative",
                       "mode_eps_150_300_3_window_30mobility_iterative",
                       "mode_eps_200_200_200_window_30mobility_iterative",
                       "mode_eps_300_300_300_window_30mobility_iterative"]

    # 11th May
    names_files_expected_11 = ["mode_eps_10_100_1_window_30_post_samples_50mobility_iterative_31",
                               "mode_eps_20_200_2_window_30_post_samples_50mobility_iterative_51",
                               "mode_eps_30_300_3_window_30_post_samples_50mobility_iterative_51",
                               "mode_eps_50_100_1_window_30_post_samples_50mobility_iterative_71",
                               "mode_eps_100_100_100_window_30_post_samples_50mobility_iterative",
                               "mode_eps_100_200_2_window_30_post_samples_50mobility_iterative_65",
                               "mode_eps_150_300_3_window_30_post_samples_50mobility_iterative_67",
                               "mode_eps_200_200_200_window_30_post_samples_50mobility_iterative_117",
                               "mode_eps_300_300_300_window_30_post_samples_50mobility_iterative_20"]
    # 23rd May
    names_files_expected_23 = ["mode_eps_10_100_1_window_30_post_samples_50mobility_iterative_23",
                               "mode_eps_20_200_2_window_30_post_samples_50mobility_iterative_48",
                               "mode_eps_30_300_3_window_30_post_samples_50mobility_iterative_49",
                               "mode_eps_50_100_1_window_30_post_samples_50mobility_iterative_30",
                               "mode_eps_100_100_100_window_30_post_samples_50mobility_iterative",
                               "mode_eps_100_200_2_window_30_post_samples_50mobility_iterative_63",
                               "mode_eps_150_300_3_window_30_post_samples_50mobility_iterative_63",
                               "mode_eps_200_200_200_window_30_post_samples_50mobility_iterative_64",
                               "mode_eps_300_300_300_window_30_post_samples_50mobility_iterative_25"]

    # for names in zip(names_files_MAP, names_files_expected_11):
    #     name_file_MAP, name_file_expected = names
    #     print(name_file_MAP, name_file_expected)
    #
    #     mobility_matrix_MAP = np.load(optimal_control_results_folder_11May + name_file_MAP + '.npy')
    #     mobility_matrix_expected = np.load(optimal_control_results_folder_11May + name_file_expected + '.npy')
    #     fig, ax = plot_mobilites_MAP_expected(T_11, mobility_matrix_MAP, mobility_matrix_expected)
    #     fig.savefig(optimal_control_results_folder_11May + name_file_MAP + ".png")
    #
    # for names in zip(names_files_MAP, names_files_expected_23):
    #     name_file_MAP, name_file_expected = names
    #     print(name_file_MAP, name_file_expected)
    #
    #     mobility_matrix_MAP = np.load(optimal_control_results_folder_23May + name_file_MAP + '.npy')
    #     mobility_matrix_expected = np.load(optimal_control_results_folder_23May + name_file_expected + '.npy')
    #     fig, ax = plot_mobilites_MAP_expected(T_23, mobility_matrix_MAP, mobility_matrix_expected)
    #     fig.savefig(optimal_control_results_folder_23May + name_file_MAP + ".png")
    #
    # for names in zip(names_files_expected_11, names_files_expected_23):
    #     name_file_11, name_file_23 = names
    #     print(name_file_11, name_file_23)
    #
    #     mobility_matrix_11 = np.load(optimal_control_results_folder_11May + name_file_11 + '.npy')
    #     mobility_matrix_23 = np.load(optimal_control_results_folder_23May + name_file_23 + '.npy')
    #     fig, ax = plot_mobilites_compare_11_23May(mobility_matrix_11, mobility_matrix_23)
    #     fig.savefig(optimal_control_results_folder_23May + "/../../comparison_11_23/expected/" + name_file_11 + ".png")
    #
    # for name in names_files_MAP:
    #     print(name)
    #
    #     mobility_matrix_11 = np.load(optimal_control_results_folder_11May + name + '.npy')
    #     mobility_matrix_23 = np.load(optimal_control_results_folder_23May + name + '.npy')
    #     fig, ax = plot_mobilites_compare_11_23May(mobility_matrix_11, mobility_matrix_23)
    #     fig.savefig(optimal_control_results_folder_23May + "/../../comparison_11_23/MAP/" + name + ".png")

    # plot that will be put in paper (11th May, 23rd May):
    ticks = [0, 9, 19, 29, 39, 49]  # , 60]
    ticklabels = ['11/05', '20/05', '30/05', '10/06', '20/06', '30/06']  # , '11/07']

    mobility_matrix_11May = np.load(optimal_control_results_folder_11May + names_files_expected_11[6] + '.npy')
    mobility_matrix_11May = mobility_matrix_11May[:, :51]  # [:,:68]  this can be used if we do not care about 11th Apr
    mobility_matrix_23May = np.load(optimal_control_results_folder_23May + names_files_expected_23[6] + '.npy')
    mobility_matrix_23May = mobility_matrix_23May[:, :39]  # [:,:56]
    fig, ax = plot_mobilites_compare_11_23May(mobility_matrix_11May, mobility_matrix_23May, ticks=ticks,
                                              ticklabels=ticklabels)

    ax.axvline(0, color="black", ls=":")
    ax.axvline(12, color="black", ls=":")

    ax.text(.5, 0.15, '11th May', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    ax.text(12.5, 0.15, '23rd May', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    fig.savefig(optimal_control_results_folder_23May + "/../../comparison/" + names_files_MAP[6] + "May_nice.pdf")

    # plot with 11th April as well
    ticks = [0, 10, 20, 30, 39, 49, 59, 69, 79]  # , 60]
    ticklabels = ['11/04', '20/04', '30/04', '11/05', '20/05', '30/05', '10/06', '20/06', '30/06']  # , '11/07']

    mobility_matrix_11Apr = np.load(
        optimal_control_results_folder_11Apr + 'mode_eps_150_300_3_window_30_post_samples_50mobility_iterative_81.npy')[
                            :, :16]
    fig, ax = plot_mobilites_compare_11Apr_11May_23May(mobility_matrix_11Apr, mobility_matrix_11May,
                                                       mobility_matrix_23May, ticks=ticks,
                                                       ticklabels=ticklabels)

    ax.axvline(0, color="black", ls=":")
    ax.axvline(30, color="black", ls=":")
    ax.axvline(42, color="black", ls=":")

    ax.text(.5, 0.21, '11th Apr', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    ax.text(30.5, 0.21, '11th May', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    ax.text(42.5, 0.21, '23rd May', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    fig.savefig(optimal_control_results_folder_23May + "/../../comparison/" + names_files_MAP[6] + "Apr_May_nice.pdf")

    # similar plot as above but for R values:
    ticks = [0, 10, 20, 30, 39, 49, 59, 69, 79]  # , 60]
    ticklabels = ['11/04', '20/04', '30/04', '11/05', '20/05', '30/05', '10/06', '20/06', '30/06']  # , '11/07']

    R_values_matrix_11Apr = np.load(
        optimal_control_results_folder_11Apr + 'mode_eps_150_300_3_window_30_post_samples_50mobility_iterative_81_R_values.npy')[
                            :, :16]
    R_values_matrix_11May = np.load(
        optimal_control_results_folder_11May + 'mode_eps_150_300_3_window_30_post_samples_50mobility_iterative_67_R_values.npy')[
                            :, :51]
    R_values_matrix_23May = np.load(
        optimal_control_results_folder_23May + 'mode_eps_150_300_3_window_30_post_samples_50mobility_iterative_63_R_values.npy')[
                            :, :39]
    fig, ax = plot_R_values_compare_11Apr_11May_23May(R_values_matrix_11Apr, R_values_matrix_11May,
                                                      R_values_matrix_23May,
                                                      ticks=ticks, ticklabels=ticklabels)

    ax.axvline(0, color="black", ls=":")
    ax.axvline(30, color="black", ls=":")
    ax.axvline(42, color="black", ls=":")
    ax.axhline(1, color="black", ls=":")

    ax.text(.5, 1.122, '11th Apr', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    ax.text(30.5, 1.122, '11th May', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    ax.text(42.5, 1.122, '23rd May', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    fig.savefig(
        optimal_control_results_folder_23May + "/../../comparison/" + names_files_MAP[6] + "Apr_May_nice_R_values.pdf")

    # both plots with 26th April values as well:
    bbox_inches = Bbox(np.array([[0.4, 0], [7.3, 3.7]]))
    savefig_kwargs = {"dpi": 150, "tightlayout": False, "bbox_inches": bbox_inches}

    mobility_matrix_26Apr = np.load(
        optimal_control_results_folder_26Apr + 'mode_eps_150_300_3_window_30_post_samples_50mobility_iterative_28.npy')[
                            :, :28]
    R_values_matrix_26Apr = np.load(
        optimal_control_results_folder_26Apr + 'mode_eps_150_300_3_window_30_post_samples_50mobility_iterative_28_R_values.npy')[
                            :, :28]
    fig, ax = plot_mobilites_compare_11Apr_26Apr_11May_23May(mobility_matrix_11Apr, mobility_matrix_26Apr,
                                                             mobility_matrix_11May, mobility_matrix_23May, ticks=ticks,
                                                             ticklabels=ticklabels)

    ax.axvline(0, color="black", ls=":")
    ax.axvline(15, color="black", ls=":")
    ax.axvline(30, color="black", ls=":")
    ax.axvline(42, color="black", ls=":")

    ax.text(.5, 0.21, '11th Apr', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    ax.text(15.5, 0.21, '26th Apr', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    ax.text(30.5, 0.21, '11th May', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    ax.text(42.5, 0.21, '23rd May', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    fig.savefig(optimal_control_results_folder_23May + "/../../comparison/" + names_files_MAP[6] + "Apr_May_nice_2.pdf",
                **savefig_kwargs)

    fig, ax = plot_R_values_compare_11Apr_26Apr_11May_23May(R_values_matrix_11Apr, R_values_matrix_26Apr,
                                                            R_values_matrix_11May, R_values_matrix_23May,
                                                            ticks=ticks, ticklabels=ticklabels)

    ax.axvline(0, color="black", ls=":")
    ax.axvline(15, color="black", ls=":")
    ax.axvline(30, color="black", ls=":")
    ax.axvline(42, color="black", ls=":")
    ax.axhline(1, color="black", ls=":")

    ax.text(.5, 1.122, '11th Apr', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    ax.text(15.5, 1.122, '26th Apr', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    ax.text(30.5, 1.122, '11th May', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    ax.text(42.5, 1.122, '23rd May', horizontalalignment='left', verticalalignment='center', alpha=.7,
            rotation=90)  # , transform=ax.transAxes)
    fig.savefig(
        optimal_control_results_folder_23May + "/../../comparison/" + names_files_MAP[
            6] + "Apr_May_nice_2_R_values.pdf", **savefig_kwargs)
