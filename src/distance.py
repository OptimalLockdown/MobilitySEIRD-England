import numpy as np

from abcpy.distances import Distance


class WeightedDistance(Distance):

    def __init__(self, distance_list, weights=None):
        """ This combines different distances and weights them
        """

        self.distance_list = distance_list
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(len(self.distance_list))

    def distance(self, d1, d2):
        """To be overwritten by any sub-class: should calculate the distance between two
        sets of data d1 and d2 using their respective statistics.

        Returns
        -------
        numpy.ndarray
            The distance between the two input data sets.
        """

        # compute the different distances:
        distances = np.array(list(map(lambda distance: distance.distance(d1, d2), self.distance_list)))
        distances *= self.weights
        return np.sum(distances)

    def dist_max(self):
        """To be overwritten by sub-class: should return maximum possible value of the
        desired distance function.

        Examples
        --------
        If the desired distance maps to :math:`\mathbb{R}`, this method should return numpy.inf.

        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """

        return np.infty
