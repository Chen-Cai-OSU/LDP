"""
tunning, featuralization, output formatting
"""

import numpy as np
import time

def set_table(graph, hyperharameter):
    table = PrettyTable();
    n_row = 5
    (hyperparameter_flag, norm_flag, loss_type, n_runs, pd_flag, multi_cv_flag, n_jobs, debug_flag, graph_isomorphisim, edge_fil, dynamic_range_flag,) = hyperparameter
    space = '  '

    title = 'hyperparameter_flag = ' + str(hyperparameter_flag) + space +\
                  '  norm_flag = ' + str(norm_flag) + space +\
                  '  loss_type = ' + str(loss_type) + space +\
                  '  n_runs = ' + str(n_runs) +  space +\
                  '  multi_cv_flag = ' + str(multi_cv_flag) + space +\
                  '  n_jobs = ' + str(n_jobs) +space +\
                  '  debug_flag =' + str(debug_flag) +space +\
                  '  graph_isomorphisim =' + str(graph_isomorphisim) +space +\
                  '  edge_fil = ' +str(edge_fil) + space +\
                  '  dynamic_range_flag = ' + str(dynamic_range_flag) + '\n'
    print(title)
    table.field_names = [graph, "DT", 'RF', "SVM/Std", 'Time/Kernel Param/PD Stat'];
    return (table, n_row)

def functionongraph(graphs_, i, key='deg', edge_flag=False):
    # for graphs_[i], get the key-val distribution

    components = len(graphs_[i]); lis = []
    for j in range(components):
        g = graphs_[i][j]
        try:
            assert (str(type(g)) ==  "<class 'networkx.classes.graphviews.SubGraph'>") or (str(type(g))) == "<class 'networkx.classes.graph.Graph'>"
        except AssertionError:
            if g is None:
                print('wired case: g is None')
                return [0]
            else:
                print('Unconsidered Cases in function on graph')

        if edge_flag==False:
            tmp = [g.nodes[k][key] for k in g.nodes]
        lis += tmp
    return lis

def hisgram_single_feature(graphs_, n_bin, key='deg', his_norm_flag='yes', edge_flag=False, lowerbound=-1, upperbound=1, cdf_flag=False, uniform_flag = True):
    import numpy as np
    n = len(graphs_)
    feature_vec = np.zeros((n, n_bin))
    for i in range(n):
        lis = functionongraph(graphs_, i, key, edge_flag=edge_flag)
        if lis == []:
            feature_vec[i] = 0
        feature_vec[i] = hisgram(lis, n_bin, his_norm_flag=his_norm_flag,
                                 lowerbound=lowerbound, upperbound=upperbound,
                                 cdf_flag=cdf_flag, uniform_flag=uniform_flag)
    return feature_vec

def hisgram(lis, n_bin=100, his_norm_flag='yes', lowerbound=-1, upperbound=1, cdf_flag=False, uniform_flag=True):
    if lis == []:
        print ('lis is empty')
        return [0]*n_bin
    # normalize lis
    # needs to be more rigirous
    # TODO: test if it helps to normalize lis
    if his_norm_flag == 'yes':
        try:
            assert max(lis) < 1.1 # * 100000 # delelte 100 later
        except AssertionError:
            print ('The max of list is %s' %max(lis)),
        assert min(lis) > -1.1
        max_ = max(lis)
        # if max_ !=0:
        #     lis = [i/float(max_) for i in lis]

    if not uniform_flag:
        assert lowerbound + 1e-3 > 0
        n_bin_ = np.logspace(np.log(lowerbound + 1e-3), np.log(upperbound),n_bin+1, base = np.e)
    else:
        n_bin_ = n_bin

    if cdf_flag == True:
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf = ECDF(lis)
        if uniform_flag:
            return ecdf([i / np.float(n_bin) for i in range(0, n_bin)])
        else:
            return ecdf([i / np.float(n_bin) for i in range(0, n_bin)])
    result = np.histogram(lis, bins=n_bin_, range=(lowerbound,upperbound))
    return result[0]

def remove_zero_col(data, cor_flag=False):
    import numpy as np
    # data = np.zeros((2,10))
    # data[1,3] = data[1,5] = data[1,7] = 1
    n_col = np.shape(data)[1]

    del_col_idx = np.where(~data.any(axis=0))[0]
    remain_col_idx = set(range(n_col)) - set(del_col_idx)
    correspondence_dict = dict(zip(range(len(remain_col_idx)), remain_col_idx))
    inverse_correspondence_dict = dict(zip(remain_col_idx, range(len(remain_col_idx))))

    X = np.delete(data, np.where(~data.any(axis=0))[0], axis=1)
    print('the shape after removing zero columns is ', np.shape(X))
    if cor_flag == True:
        return (X, correspondence_dict, inverse_correspondence_dict)
    else:
        return X


def merge_features(graph, graphs_, allowed, n_bin=30, his_norm_flag='yes', edge_flag=False, cdf_flag=False, uniform_flag = True):
    print('Number of bins are %s'%n_bin)
    n = len(graphs_)
    X = np.zeros((n, 1))
    for key in allowed:
        # print(key)
        if (key=='label') :
            if graph == 'dd_test':
                nbin = 90
            else:
                nbin = 40
            tmp = hisgram_single_feature(graphs_, nbin, 'label', his_norm_flag=his_norm_flag, edge_flag=edge_flag, lowerbound=0, upperbound=1, cdf_flag=cdf_flag, uniform_flag=uniform_flag)

        elif key == 'ricciCurvature': # use default bound for ricci curvature
            tmp = hisgram_single_feature(graphs_, n_bin, key, his_norm_flag=his_norm_flag, edge_flag=edge_flag, cdf_flag=cdf_flag, uniform_flag=uniform_flag)
        else:
            tmp = hisgram_single_feature(graphs_, n_bin, key, his_norm_flag=his_norm_flag, edge_flag=edge_flag, cdf_flag=cdf_flag, uniform_flag=uniform_flag, lowerbound=0)
        X = np.append(X, tmp, axis=1)
    return remove_zero_col(X[:,1:])


def get_i_dist_distribution(i, print_flag=False, cdf_flag=False):
    t1 = time.time()
    gs = graphs_[i]
    result = np.zeros((1, 30))
    for g in gs:
        result += dist_distribution(g, cdf_flag=cdf_flag)
    print('-'),
    try:
        if print_flag:
            print(i, len(gs[0]), len(gs[0].edges()),time.time()-t1)
    except:
        print('Graphs %s has some problem'%i)
    return result

# @cached('distance_distribution')
def get_dist_distribution(n_jobs=-1, print_flag=False, cdf_flag=False):
    import json
    n = len(graphs_)
    from joblib import Parallel, delayed
    total = Parallel(n_jobs=n_jobs, verbose=5)(delayed(get_i_dist_distribution)(i, print_flag=print_flag, cdf_flag=cdf_flag) for i in range(n))
    total = np.array(total)
    feature = np.stack(total, axis=1)[0]
    return feature

def dist_distribution(g, cdf_flag=False):
    # g = graphs_[0][0]
    if g == None:
        return np.zeros((1, 30))
    assert nx.is_connected(g)
    dist_dict = dict(nx.all_pairs_shortest_path_length(g))
    dist_distribution_tmp = [dist_dict[i].values() for i in dist_dict.keys()]
    dist_distribution_ = [val for sublist in dist_distribution_tmp for val in sublist]
    # for v1 in g.nodes():
    #     for v2 in g.nodes():
    #         dist_distrbution += [dist_dict[v1][v2]]

    assert len(dist_distribution_) == len(g) ** 2
    if cdf_flag == True:
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf = ECDF(dist_distribution_)
        return ecdf(range(0, 30))
    return np.histogram(dist_distribution_, range(31))[0]
