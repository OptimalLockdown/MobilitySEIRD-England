import logging

import numpy as np
from abcpy.backends import BackendDummy as Backend
from abcpy.continuousmodels import Uniform
# from abcpy.backends import BackendMPI as Backend  # to use MPI
from abcpy.distances import Euclidean
from abcpy.output import Journal

from src.distance import WeightedDistance
from src.models import SEI4RD
from src.statistic import ExtractSingleTimeseries2DArray
from src.utils import ABC_inference, determine_eps, generate_samples

logging.basicConfig(level=logging.INFO)
#results_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_23May/"
results_folder = "results/PMCABC/real_data/SEI4RD_england_infer_1Mar_31Aug/"
print(results_folder)
# load files

#data_folder = "data/england_inference_data_1Mar_to_23May/"
data_folder = "data/england_inference_data_1Mar_to_31Aug/"

# alpha_home = np.load(data_folder + "alpha_home.npy")
alpha_home = 1  # fix this
alpha_work = np.load(data_folder + "mobility_work.npy")
alpha_other = np.load(data_folder + "mobility_other.npy")
alpha_school = np.load(data_folder + "mobility_school.npy")

england_pop = np.load(data_folder + "england_pop.npy", allow_pickle=True)

contact_matrix_home_england = np.load(data_folder + "contact_matrix_home_england.npy")
contact_matrix_work_england = np.load(data_folder + "contact_matrix_work_england.npy")
contact_matrix_school_england = np.load(data_folder + "contact_matrix_school_england.npy")
contact_matrix_other_england = np.load(data_folder + "contact_matrix_other_england.npy")

observation_england = np.load(data_folder + "observed_data.npy", allow_pickle=True)
#print(observation_england.shape)
# thee last column of the above one represents the number of Ic people in our model.

# parameters
n = 5  # number of age groups
dt = 0.1  # integration timestep
T = alpha_other.shape[0] - 1  # horizon time in days (needs to be 1 less than the number of days in observation)
total_population = england_pop  # population for each age group
# 16th March: Boris Johnson asked old people to isolate; we then learn a new alpha from the 18th March:
lockdown_day = 17

# alpha_home = np.repeat(alpha_home, np.int(1 / dt), axis=0)
alpha_work = np.repeat(alpha_work, np.int(1 / dt), axis=0)
alpha_other = np.repeat(alpha_other, np.int(1 / dt), axis=0)
alpha_school = np.repeat(alpha_school, np.int(1 / dt), axis=0)

# ABC model (priors need to be fixed better):
# beta = Uniform([[0], [1]], name='beta')  # controls how fast the epidemics grows. Related to R_0
beta = Uniform([[0], [0.5]], name='beta')  # controls how fast the epidemics grows. Related to R_0
d_L = Uniform([[1], [10]], name='d_L')  # average duration of incubation
d_C = Uniform([[1], [10]], name='d_C')  # average time before going to clinical
d_R = Uniform([[1], [10]], name='d_R')  # average recovery time
d_RC = Uniform([[4], [14]], name='d_RC')  # average recovery time from clinical state
d_D = Uniform([[1], [10]], name='d_D')  # average duration of infected clinical state (resulting in death)
p01 = Uniform([[0], [.3]], name="p01")  # restrict priors a bit
p02 = Uniform([[0], [.5]], name="p02")
p03 = Uniform([[0], [1]], name="p03")
p04 = Uniform([[0], [1]], name="p04")
p05 = Uniform([[0.5], [1]], name="p05")
p11 = Uniform([[0], [1]], name="p11")
p12 = Uniform([[0], [1]], name="p12")
p13 = Uniform([[0], [1]], name="p13")
p14 = Uniform([[0], [1]], name="p14")
p15 = Uniform([[0.5], [1]], name="p15")
initial_exposed = Uniform([[0], [500]], name="initial_exposed")
alpha_123 = Uniform([[0.3], [1]], name="alpha_123")
alpha_4 = Uniform([[0], [1]], name="alpha_4")
alpha_5 = Uniform([[0], [1]], name="alpha_5")

model = SEI4RD(
    [beta, d_L, d_C, d_R, d_RC, d_D, p01, p02, p03, p04, p05, p11, p12, p13, p14, p15, initial_exposed, alpha_123,
     alpha_4, alpha_5], tot_population=total_population, T=T, contact_matrix_school=contact_matrix_school_england,
    contact_matrix_work=contact_matrix_work_england, contact_matrix_home=contact_matrix_home_england,
    contact_matrix_other=contact_matrix_other_england, alpha_school=alpha_school, alpha_work=alpha_work,
    alpha_home=alpha_home, alpha_other=alpha_other, modify_alpha_home=False, dt=dt, return_once_a_day=True,
    return_observation_only_with_hospitalized=True, learn_alphas_old=True, lockdown_day=lockdown_day)
true_parameters_fake_1 = [0.05, 5, 7, 5, 5, 6, 0.06, .1, .2, .3, .4, .1, .2, .3, .4, .5, 50, .4, .3, 0.3]
# print(len(true_parameters_fake_1))
observation_england_1 = model.forward_simulate(true_parameters_fake_1, 1)
true_parameters_fake_2 = [0.05, 5, 7, 5, 5, 6, 0.05, .1, .2, .3, .4, .1, .2, .3, .4, .5, 50, .4, .3, 0.3]
observation_england_2 = model.forward_simulate(true_parameters_fake_2, 1)

print(observation_england_1[0].shape)

# we define now the statistics and distances:
rho = 1  # multiplier to decrease importance of past
distances_list = []

# this has to be used if the model returns already the correct observations (return_observation_only=True)
for i in range(n):
    distances_list.append(
        Euclidean(ExtractSingleTimeseries2DArray(index=i, rho=rho, end_step=-1)))  # deceased
# now add the distance on the number of hospitalized people, that needs to discard the first 17 elements because there
# is no data on that before the 18th March.
distances_list.append(Euclidean(ExtractSingleTimeseries2DArray(index=5, rho=rho, start_step=19, end_step=-1)))

# define a weighted distance:
# max values of the daily counts:  1.,   9.,  73., 354., 462., 17933.
# we could use the inverse of them as weights; I think however the last timeseries have less noise as they are sampled 
# from larger numbers, so they should be slightly less important.
weights = [1, 1, 1, 2, 2, .1]
# weights = [1.0 / 1 * 0.75, 1.0 / 9 * 0.75, 1.0 / 68 * 0.85, 1.0 / 338, 1.0 / 445, 1.0 / 4426]
# weights = [1, 0.1, 0.01, 0.005, 0.005, 0.005]
final_distance = WeightedDistance(distances_list, weights=weights)
print("dist", final_distance.distance(observation_england_1, [observation_england]))

# define backend
backend = Backend()

# # generate 100 samples from which to find the starting epsilon value as the 20th percentile of the distances
param, samples = generate_samples(model, 10, num_timeseries=6, two_dimensional=True)
eps = determine_eps(samples, dist_calc=final_distance, quantile=0.5)  # * 1000

print("epsilon", eps)

# you can keep running the sequential algorithms from previously saved journal files.
start_journal_path = None
# start_journal_path = results_folder + "seicicsr.jrl"
# jrnl_start = Journal.fromFile(start_journal_path)
# eps = jrnl_start.configuration["epsilon_arr"][-1]  # use the last step eps? Maybe should reduce that as well..

# inference1
print("Inference 1")
jrnl = ABC_inference("PMCABC", model, observation_england, final_distance, eps=eps, n_samples=500, n_steps=5,
                     backend=backend, full_output=1, journal_file=start_journal_path, epsilon_percentile=10,
                     journal_file_save=results_folder + "journal_1")
# jrnl = ABC_inference("SABC", model, observation_england, final_distance, eps=eps, n_samples=100000, n_steps=10,
#                      backend=backend, full_output=1, journal_file=start_journal_path, beta=2,
#                      delta=0.2, v=0.3, ar_cutoff=0.01, resample=None, n_update=None, )
jrnl.save(results_folder + "PMCABC_inf1.jrl")

print("Posterior mean: ", jrnl.posterior_mean())
#
# inference2
print("Inference 2")
epsilon_percentile = 50

if "epsilon_arr" in jrnl.configuration.keys():
    eps = np.percentile(jrnl.distances[-1], epsilon_percentile)
    print("using epsilon from last step...")

start_journal_path = results_folder + "PMCABC_inf1.jrl"
jrnl = ABC_inference("PMCABC", model, observation_england, final_distance, eps=eps, n_samples=500, n_steps=10,
                     backend=backend, full_output=1, journal_file=start_journal_path,
                     epsilon_percentile=epsilon_percentile,
                     journal_file_save=results_folder + "journal_2")

# save the journal
jrnl.save(results_folder + "PMCABC_inf2.jrl")
#
print("Inference 3")
jrnl = Journal.fromFile(results_folder + "journal_2.jrl")
epsilon_percentile = 70

if "epsilon_arr" in jrnl.configuration.keys():
    eps = np.percentile(jrnl.distances[-1], epsilon_percentile)
    print("using epsilon from last step: ", eps)

start_journal_path = results_folder + "journal_2.jrl"
jrnl = ABC_inference("PMCABC", model, observation_england, final_distance, eps=eps, n_samples=500, n_steps=10,
                     backend=backend, full_output=1, journal_file=start_journal_path,
                     epsilon_percentile=epsilon_percentile,
                     journal_file_save=results_folder + "journal_3")

# save the journal
jrnl.save(results_folder + "PMCABC_inf3.jrl")
