import copy
import logging

from abcpy.acceptedparametersmanager import *
from abcpy.inferences import InferenceMethod, BaseDiscrepancy
from abcpy.jointdistances import LinearCombination
from abcpy.output import Journal
from abcpy.perturbationkernel import DefaultKernel
from abcpy.probabilisticmodels import *


# we modify the PMCABC algorithm in the ABCpy package so that it saves the journal file at each iteration

class PMCABC(BaseDiscrepancy, InferenceMethod):
    """
    This base class implements a modified version of Population Monte Carlo based inference scheme for Approximate
    Bayesian computation of Beaumont et. al. [1]. Here the threshold value at `t`-th generation are adaptively chosen by
    taking the maximum between the epsilon_percentile-th value of discrepancies of the accepted parameters at `t-1`-th
    generation and the threshold value provided for this generation by the user. If we take the value of
    epsilon_percentile to be zero (default), this method becomes the inference scheme described in [1], where the
    threshold values considered at each generation are the ones provided by the user.

    [1] M. A. Beaumont. Approximate Bayesian computation in evolution and ecology. Annual Review of Ecology,
    Evolution, and Systematics, 41(1):379–406, Nov. 2010.

    Parameters
    ----------
    model : list
        A list of the Probabilistic models corresponding to the observed datasets
    distance : abcpy.distances.Distance
        Distance object defining the distance measure to compare simulated and observed data sets.
    kernel : abcpy.distributions.Distribution
        Distribution object defining the perturbation kernel needed for the sampling.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    seed : integer, optional
         Optional initial seed for the random number generator. The default value is generated randomly.
    """

    model = None
    distance = None
    kernel = None
    rng = None

    # default value, set so that testing works
    n_samples = 2
    n_samples_per_param = None

    backend = None

    def __init__(self, root_models, distances, backend, kernel=None, seed=None):
        self.model = root_models
        # We define the joint Linear combination distance using all the distances for each individual models
        self.distance = LinearCombination(root_models, distances)
        if (kernel is None):

            mapping, garbage_index = self._get_mapping()
            models = []
            for mdl, mdl_index in mapping:
                models.append(mdl)
            kernel = DefaultKernel(models)

        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger(__name__)

        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

        self.simulation_counter = 0

    def sample(self, observations, steps, epsilon_init, n_samples=10000, n_samples_per_param=1, epsilon_percentile=10,
               covFactor=2, full_output=0, journal_file=None, journal_file_save=None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets
        steps : integer
            Number of iterations in the sequential algoritm ("generations")
        epsilon_init : numpy.ndarray
            An array of proposed values of epsilon to be used at each steps. Can be supplied
            A single value to be used as the threshold in Step 1 or a `steps`-dimensional array of values to be
            used as the threshold in evry steps.
        n_samples : integer, optional
            Number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        epsilon_percentile : float, optional
            A value between [0, 100]. The default value is 10.
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default value is 2 as considered in [1].
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """
        self.accepted_parameters_manager.broadcast(self.backend, observations)
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        if (journal_file is None):
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = type(self.distance).__name__
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["steps"] = steps
            journal.configuration["epsilon_percentile"] = epsilon_percentile
        else:
            journal = Journal.fromFile(journal_file)

        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mats = None

        # Define epsilon_arr
        if len(epsilon_init) == steps:
            epsilon_arr = epsilon_init
        else:
            if len(epsilon_init) == 1:
                epsilon_arr = [None] * steps
                epsilon_arr[0] = epsilon_init
            else:
                raise ValueError("The length of epsilon_init can only be equal to 1 or steps.")

        # main PMCABC algorithm
        self.logger.info("Starting PMC iterations")
        for aStep in range(steps):
            self.logger.debug("iteration {} of PMC algorithm".format(aStep))
            if (aStep == 0 and journal_file is not None):
                accepted_parameters = journal.get_accepted_parameters(-1)
                accepted_weights = journal.get_weights(-1)

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                # 3: calculate covariance
                self.logger.info("Calculateing covariance matrix")
                new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                # Since each entry of new_cov_mats is a numpy array, we can multiply like this
                accepted_cov_mats = [covFactor * new_cov_mat for new_cov_mat in new_cov_mats]

            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            rng_pds = self.backend.parallelize(rng_arr)

            # 0: update remotely required variables
            # print("INFO: Broadcasting parameters.")
            self.logger.info("Broadcasting parameters")
            self.epsilon = epsilon_arr[aStep]
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters, accepted_weights,
                                                              accepted_cov_mats)

            # 1: calculate resample parameters
            # print("INFO: Resampling parameters")
            self.logger.info("Resampling parameters")

            params_and_dists_and_counter_pds = self.backend.map(self._resample_parameter, rng_pds)
            params_and_dists_and_counter = self.backend.collect(params_and_dists_and_counter_pds)
            new_parameters, distances, counter = [list(t) for t in zip(*params_and_dists_and_counter)]
            new_parameters = np.array(new_parameters)
            distances = np.array(distances)

            for count in counter:
                self.simulation_counter += count

            # Compute epsilon for next step
            # print("INFO: Calculating acceptance threshold (epsilon).")
            self.logger.info("Calculating acceptances threshold")
            if aStep < steps - 1:
                if epsilon_arr[aStep + 1] == None:
                    epsilon_arr[aStep + 1] = np.percentile(distances, epsilon_percentile)
                else:
                    epsilon_arr[aStep + 1] = np.max(
                        [np.percentile(distances, epsilon_percentile), epsilon_arr[aStep + 1]])

            # 2: calculate weights for new parameters
            self.logger.info("Calculating weights")

            new_parameters_pds = self.backend.parallelize(new_parameters)
            self.logger.info("Calculate weights")
            new_weights_pds = self.backend.map(self._calculate_weight, new_parameters_pds)
            new_weights = np.array(self.backend.collect(new_weights_pds)).reshape(-1, 1)
            sum_of_weights = 0.0
            for w in new_weights:
                sum_of_weights += w
            new_weights = new_weights / sum_of_weights

            # The calculation of cov_mats needs the new weights and new parameters
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=new_parameters,
                                                              accepted_weights=new_weights)

            # The parameters relevant to each kernel have to be used to calculate n_sample times. It is therefore more efficient to broadcast these parameters once,
            # instead of collecting them at each kernel in each step
            kernel_parameters = []
            for kernel in self.kernel.kernels:
                kernel_parameters.append(
                    self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
            self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

            # 3: calculate covariance
            self.logger.info("Calculating covariance matrix")
            new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
            # Since each entry of new_cov_mats is a numpy array, we can multiply like this
            new_cov_mats = [covFactor * new_cov_mat for new_cov_mat in new_cov_mats]

            # 4: Update the newly computed values
            accepted_parameters = new_parameters
            accepted_weights = new_weights
            accepted_cov_mats = new_cov_mats

            self.logger.info("Save configuration to output journal")

            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_distances(copy.deepcopy(distances))
                journal.add_weights(copy.deepcopy(accepted_weights))
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)
                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)
                print(journal_file_save)
                if journal_file_save is not None:
                    if full_output == 1:
                        journal.save(journal_file_save + '.jrl')  # avoid writing a lot of different files.
                    else:
                        journal.save(journal_file_save + '_' + str(aStep) + '.jrl')

        # Add epsilon_arr to the journal
        journal.configuration["epsilon_arr"] = epsilon_arr

        return journal

    def _resample_parameter(self, rng, npc=None):
        """
        Samples a single model parameter and simulate from it until
        distance between simulated outcome and the observation is
        smaller than epsilon.

        Parameters
        ----------
        seed: integer
            initial seed for the random number generator.

        Returns
        -------
        np.array
            accepted parameter
        """

        # print(npc.communicator())
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        distance = self.distance.dist_max()

        if distance < self.epsilon and self.logger:
            self.logger.warn("initial epsilon {:e} is larger than dist_max {:e}"
                             .format(float(self.epsilon), distance))

        theta = self.get_parameters()
        counter = 0

        while distance > self.epsilon:
            if self.accepted_parameters_manager.accepted_parameters_bds == None:
                self.sample_from_prior(rng=rng)
                theta = self.get_parameters()
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter += 1

            else:
                index = rng.choice(self.n_samples, size=1,
                                   p=self.accepted_parameters_manager.accepted_weights_bds.value().reshape(-1))
                # truncate the normal to the bounds of parameter space of the model
                # truncating the normal like this is fine: https://arxiv.org/pdf/0907.4010v1.pdf
                while True:
                    perturbation_output = self.perturb(index[0], rng=rng)
                    if (perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0):
                        theta = perturbation_output[1]
                        break
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter += 1

            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)

            self.logger.debug("distance after {:4d} simulations: {:e}".format(
                counter, distance))

        self.logger.debug(
            "Needed {:4d} simulations to reach distance {:e} < epsilon = {:e}".
                format(counter, distance, float(self.epsilon))
        )

        return (theta, distance, counter)

    def _calculate_weight(self, theta, npc=None):
        """
        Calculates the weight for the given parameter using
        accepted_parameters, accepted_cov_mat

        Parameters
        ----------
        theta: np.array
            1xp matrix containing model parameter, where p is the number of parameters

        Returns
        -------
        float
            the new weight for theta
        """
        self.logger.debug("_calculate_weight")
        if self.accepted_parameters_manager.kernel_parameters_bds is None:
            return 1.0 / self.n_samples
        else:
            prior_prob = self.pdf_of_prior(self.model, theta, 0)

            denominator = 0.0

            # Get the mapping of the models to be used by the kernels
            mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(
                self.accepted_parameters_manager.model)

            for i in range(0, self.n_samples):
                pdf_value = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                            self.accepted_parameters_manager.accepted_parameters_bds.value()[i], theta)
                denominator += self.accepted_parameters_manager.accepted_weights_bds.value()[i, 0] * pdf_value
            return 1.0 * prior_prob / denominator
