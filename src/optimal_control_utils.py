import numpy as np

from src.utils import stack_alphas


def obtain_alphas_from_mobility(alpha_123, alpha_4, alpha_5, mobility_school, mobility_work, mobility_other):
    alpha_123_timeseries = np.ones_like(mobility_school) * alpha_123
    alpha_4_timeseries = np.ones_like(mobility_school) * alpha_4
    alpha_5_timeseries = np.ones_like(mobility_school) * alpha_5

    # now we stack with the other alphas
    alpha_school = stack_alphas(alpha_123_timeseries * mobility_school, alpha_4_timeseries,
                                alpha_5_timeseries)
    alpha_work = stack_alphas(alpha_123_timeseries * mobility_work, alpha_4_timeseries,
                              alpha_5_timeseries)
    alpha_other = stack_alphas(alpha_123_timeseries * mobility_other, alpha_4_timeseries,
                               alpha_5_timeseries)
    return alpha_school, alpha_work, alpha_other


def bound_start_constrained(n_days, end_training_alpha_values):
    bounds, ind2 = [], 0
    for ind in range(3 * n_days):  # 3 instead of 4 as we are not using alpha_home
        if ind == (ind2 * n_days):
            # if ind2 == 2:
            #     bounds_current_alpha = (.85, .95)  # ?
            # else:
            bounds_current_alpha = (end_training_alpha_values[ind2] - .05, end_training_alpha_values[ind2] + .05)
            bounds.append(bounds_current_alpha)
            ind2 += 1
        else:
            bounds.append((.1, 1))
    return bounds


def realistic(alpha_initial, n_days):
    bounds = []
    # school
    for ind in range(n_days):
        bounds.append((.1, 1))
    # work
    for ind in range(n_days):
        bounds.append((0.31692307692307675, 1))
    # other
    for ind in range(n_days):
        bounds.append((0.41472727272727283, 1))

    # for ind in range(len(alpha_initial)):
    #    if alpha_initial[ind] > 1:
    #        bounds.append((1, alpha_initial[ind]))
    #    else:
    #        bounds.append((alpha_initial[ind], 1))
    #        # bounds.append((0, 1))
    return bounds


def different_bounds(argument, n_days, alpha_initial=None, end_training_alpha_values=None):
    switcher = {
        'startconstrained': bound_start_constrained(n_days, end_training_alpha_values),
        'realistic': realistic(alpha_initial, n_days),
    }
    # what is going on here?
    return switcher.get(argument, [(0, 1) for ind in range(3 * n_days)])
