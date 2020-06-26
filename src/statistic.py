import numpy as np

from abcpy.statistics import Statistics


class ExtractSingleTimeseries2DArray(Statistics):
    """
    This has to be used with the SEI4RD model.
    """

    def __init__(self, index, start_step=0, end_step=-1, degree=1, cross=False, rho=1, previous_statistics=None):
        """
        Parameters
        ----------
        degree : integer, optional
            Of polynomial expansion. The default value is 2 meaning second order polynomial expansion.
        cross : boolean, optional
            Defines whether to include the cross-product terms. The default value is True, meaning the cross product term
            is included.
        previous_statistics : Statistics class, optional
            It allows pipelining of Statistics. Specifically, if the final statistic to be used is determined by the
            composition of two Statistics, you can pass the first here; then, whenever the final statistic is needed, it
            is sufficient to call the `statistics` method of the second one, and that will automatically apply both
            transformations.
        """
        self.index = index
        self.degree = degree
        self.cross = cross
        self.previous_statistics = previous_statistics
        self.rho = rho
        self.start_step = start_step
        self.end_step = end_step

    def statistics(self, data):
        """
        Parameters
        ----------
        data: python list
            Contains n data sets with length p.
        Returns
        -------
        numpy.ndarray
            nx(p+degree*p+cross*nchoosek(p,2)) matrix where for each of the n data points with length p,
            (p+degree*p+cross*nchoosek(p,2)) statistics are calculated.
        """
        # remove last trajectory (as the 5 trajectories sum to a constant)
        # note that if you do an inplace modification, the ABC inference does not work!
        new_data = []
        for i in range(len(data)):
            timeseries = data[i][self.start_step:self.end_step, self.index]
            T = timeseries.shape[0]
            if self.rho < 1:
                timeseries *= self.rho ** (T - 1 - np.arange(T))
            new_data.append(timeseries)

        if self.previous_statistics is not None:
            data = self.previous_statistics.statistics(new_data)
        # the first of the statistics need to take list as input, in order to match the API. Then actually the
        # transformations work on np.arrays. In fact the first statistic transforms the list to array. Therefore, the
        # following code needs to be called only if the self statistic is the first, i.e. it does not have a
        # previous_statistic element.
        else:
            data = self._check_and_transform_input(new_data)

        # Expand the data with polynomial expansion
        result = self._polynomial_expansion(data)

        return result
