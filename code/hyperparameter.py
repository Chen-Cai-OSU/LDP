def load_best_params_(dataset):
    # the following parameters are used to replicate the results for linear kernel

    if dataset == 'imdb_binary':
        best_params_ = {'kernel': 'linear', 'C': 100}
        others = {'n_bin': 70, 'norm_flag': 'no', 'cdf_flag': True }
        res = 74.0
        note = {'without using any normalization'}
    elif dataset == 'imdb_multi':
        best_params_ = {'kernel': 'linear', 'C': 10}
        others = {'n_bin': 100, 'norm_flag': 'no', 'cdf_flag': True }
        res = 49.8
        note = {'without using any normalization'}
    elif dataset == 'reddit_binary':
        best_params_ = {'kernel': 'linear', 'C': 1000}
        others = {'n_bin': 100, 'norm_flag': 'no', 'cdf_flag': True}
        res = 90.0
        note = {'normalize at axis 1'}
    elif dataset == 'collab':
        best_params_ = {'kernel': 'linear', 'C': 100}
        others = {'n_bin': 100, 'norm_flag': 'no', 'cdf_flag': True}
        res = 74.5
        note = {'without using any normalization'}
    elif dataset == 'reddit_5K':
        best_params_ = {'kernel': 'linear', 'C': 10000}
        others = {'n_bin': 50, 'norm_flag': 'yes', 'cdf_flag': True, 'uniform_flag': False}
        res = 54.7
        note = {'normalize at axis 1'}
    elif dataset == 'reddit_12K':
        best_params_ = {'kernel': 'linear', 'C': 1000}
        others = {'n_bin': 70, 'norm_flag': 'yes', 'cdf_flag': True, 'uniform_flag': False}
        res = 46.7
        note = {'normalize at axis 1'}
    else:
        raise Exception('No such dataset')

    return best_params_
