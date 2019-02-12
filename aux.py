import random; random.seed(42)
import sys
import networkx as nx
import numpy as np
import argparse
sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode')
sys.path.append('/home/cai.507/Documents/Utilities/dionysus/build/bindings/python')
import dionysus as d
from cycle_tools import *
from helper import *
from localtest import *
from test0 import high_order
from testground import aggregation_feature, searchclf, evaluate_clf
from joblib import Parallel, delayed
from profilehooks import profile

def pipeline0(i, allowed, version=1, print_flag='False', norm_flag = 'no', feature_addition_flag =False):
    # basically two steps. 1) convert data dict to netowrkx graph 2) calculate function on networkx graphs
    # bar.next()
    print('*'),
    # version1: deal with chemical graphs
    # version2: deal with all non-isomorphic graphs
    import os
    # prepare data. Only execute once.
    if version==1:
        assert 'data' in globals()
        if not feature_addition_flag:
            gi = convert2nx(data[i], i)

    if not feature_addition_flag:
        subgraphs = get_subgraphs(gi)
    elif feature_addition_flag:
        assert 'graphs_' in globals().keys()
        # global graphs_
        subgraphs = [g.copy() for g in graphs_[i]]

    gi_s = [function_basis(gi, allowed, norm_flag=norm_flag) for gi in subgraphs]
    gi_s = [g for g in gi_s if g!=None]
    if print_flag == 'True':
        pass
        # print('graph %s, n_nodes: %s, n_edges: %s',%(i,len(gi_s[0]),len(gi_s[0].edge)))
    # print('OS: %s, Graph %s: Pipeline1 Finishes'%(os.getpid(), i))
    return gi_s

def pipeline1(i, beta=np.array([0,0,0,0,1]), hop_flag = 'n', basep = 0, debug='off', rs=100, edge_fil='off'): # beta= [deg, ricci, fiedler, cc]
    # data: mutag dict
    # calculate persistence diagram of graph(may disconneced)
    import dionysus as d
    if (i%50==0):
        print('.'),
    if debug == 'on':
        print('Processing %s' % i)
    assert 'data' in globals()
    dgm_ = d.Diagram()
    subgraphs = []; dgm_= d.Diagram([(0,0)]); dgm_sub=d.Diagram([(0,0)]); dgm_super=d.Diagram([(0,0)]); epd_dgm=d.Diagram([(0,0)])
    for k in range(len(graphs_[i])):
        if debug == 'on':
            print('Processing subgraph %s'%k)

        g = graphs_[i][k]
        assert str(type(g)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"
        g = fv(g, beta, hop_flag=hop_flag, basep = basep, rs=rs, edge_fil=edge_fil)  # belong to pipe1
        (g, fv_list) = add_function_value(g, fv_input='fv_test', edge_value='max')  # belong to pipe1
        dgm_sub = get_diagram(g, key='fv', subflag='True')

        (g, fv_list) = add_function_value(g, fv_input='fv_test', edge_value='min')  # belong to pipe1
        dgm_super = get_diagram(g, key='fv', subflag='False')
        dgm_super = flip_dgm(dgm_super)
        epd_dgm = get_diagram(g, key='fv', one_homology_flag=True)

        dgm = add_dgms(dgm_sub, dgm_super)
        if debug == 'on':
            print('Individual dgm:'),
            print_dgm(dgm)
        dgm_ = add_dgms(dgm_, dgm)
        subgraphs.append(g)

    if debug == 'on':
        print('Final dgm:'),
        print_dgm(dgm_)
    if i%100==0:
        print_dgm(dgm)
    return (subgraphs, dgm_, dgm_sub, dgm_super, epd_dgm)

def getbaselineX(i):
    from testground import attributes
    assert 'graphs_' in globals()
    g = graphs_[i][0]
    return attributes(g)

def f_transform(x, param={}, type_='poly'):
    # transform filtration function x -> f(x) where f can be polynomial
    if type == 'log':
        import numpy as np
        return np.log(x+1)
    if type_=='poly':
        assert type(param)==dict
        return param['a']* x**2 + param['b'] * x + param['c']
    if type_=='identity':
        return x


def add_function_value(gi, fv_input='fv_test', edge_value='max'):
    # gi is nx graph with cc, deg, fiedler, hop, lebael, ricci computed
    # add fv function value for edges
    # fv_input here is use the existing fv as fv

    import numpy as np
    import networkx as nx
    import random
    # deg = dict(gi.degree())
    # closeness_centrality_dict = nx.closeness_centrality(gi)
    # for n in closeness_centrality_dict.keys():
    #     closeness_centrality_dict[n] = 1 / closeness_centrality_dict[n]

    fv_test = {i: gi.nodes[i]['fv_test']  for i in gi.nodes()} # the key is not necessarily consecutive

    # legacy code
    if fv_input == 'deg':
        pass
        # fv = deg
    elif fv_input == 'closeness_centrality':
        pass
        # fv = closeness_centrality_dict
    elif fv_input == 'fv_test':
        fv = fv_test

    assert type(fv) == dict
    # fv_list.sort()
    fv_random = {}
    for i in fv.keys():
        fv_random[i] = fv[i] + random.uniform(0, 1e-8)
    # assert len(np.unique(fv_list)) == len(fv_list)
    # needs to consider non-consecutive case

    for i in gi.node():
        # gi.node[i]['deg'] = deg[i]
        # gi.node[i]['closeness_centrality'] = closeness_centrality_dict[i]
        gi.node[i]['fv'] = fv[i]
        gi.node[i]['fv_random'] = fv_random[i]

    for (e1, e2) in gi.edges():
        if edge_value == 'max':
            gi[e1][e2]['fv'] = max(gi.node[e1]['fv'], gi.node[e2]['fv'])
            gi[e1][e2]['fv_random'] = max(gi.node[e1]['fv_random'], gi.node[e2]['fv_random'])
        if edge_value == 'min':
            gi[e1][e2]['fv'] = min(gi.node[e1]['fv'], gi.node[e2]['fv'])
            gi[e1][e2]['fv_random'] = min(gi.node[e1]['fv_random'], gi.node[e2]['fv_random'])
        assert type(fv_random) == dict
    tmp = fv_random.values()
    tmp.sort()
    return (gi, tmp)
    # return (gi, fv_list)

def baseline(X, Y, graphs_, allowed, n_splits=10, multi_cv_flag=False, skip_rf=False, laplacian_flag=False, lap_band=10, skip_svm=False):
    import time
    if not skip_rf:
        rfresult =  rfclf(X, Y, m_f='auto', multi_cv_flag=multi_cv_flag)
    elif skip_rf:
        rfresult = [0,0]
    rfresult = ["{0:.1f}".format(100 * i) for i in rfresult]

    if skip_svm:
        return (rfresult[0], rfresult[1], '0', '0', '0', '0')

    time1 = time.time()
    print('Using deg function as a baseline')
    if  graph == 'reddit_12K':
        nonlinear_flag = 'reddit_12K'
        param = searchclf(X, Y, 1001, test_size=0.1, nonlinear_flag=nonlinear_flag, verbose=0, laplacian_flag=laplacian_flag, lap_band=lap_band, print_flag='on')
        # param = {'kernel': 'linear', 'C': 1000}
    else:
        nonlinear_flag = 'True'
        param = searchclf(X, Y, 1001, test_size=0.1, nonlinear_flag=nonlinear_flag, verbose=0, laplacian_flag=laplacian_flag, lap_band=lap_band)
    svm_result = evaluate_clf(graph, X, Y, param, n_splits=n_splits)
    svm_result = ["{0:.1f}".format(100 * i) for i in svm_result]
    time2 = time.time()

    return (rfresult[0], rfresult[1], svm_result[0], svm_result[1], round(time2-time1), str(param))

def readargs():
    import os, sys
    import numpy as np
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', help='The type of graph', default='mutag')
    parser.add_argument('-rf','--rf_flag', help='y if only use rf', default='n')
    parser.add_argument('-b', '--beta', help='The combination of deg, ricci, fiedler, cc and label',
                        default=np.array([1, 1, 0, 0, 0]), type=np.ndarray)
    parser.add_argument('-norm_flag', '--norm_flag', help='yes means normalize when computing function, no means no normalization(default)',
                        default='no')
    parser.add_argument('-high_order_flag', '--high_order_flag', help='Decide whether to use high order graphs', default=False)
    parser.add_argument('-p', '--parameter', help='The parameter of filtration function', default={'a':0, 'b':1, 'c':0}, type=json.loads)
    parser.add_argument('-pa', '--pa', help='The parameter of filtration function', default=0, type=float)
    parser.add_argument('-pb', '--pb', help='The parameter of filtration function', default=1, type=float)

    # parser.add_argument('-kp', "--keypairs", dest="my_dict", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL")
    args = parser.parse_args()

    if len(sys.argv) != 1:
        graph = args.graph
        beta = args.beta
        rf_flag = args.rf_flag
        norm_flag = args.norm_flag
        high_order_flag = args.high_order_flag
        parameter_flag = args.parameter
        pa = args.pa; pb = args.pb
    else:
        graph = 'mutag'
        beta = np.array([1, 1, 0, 0, 0])
        beta = beta / np.float(np.sum(beta))
        parameter_flag = {'a':0, 'b':1, 'c':0}
        # keypairs = {'a':0, 'b':1, 'c':0}
        pa = 0; pb = 1;
        assert abs(np.sum(beta) - 1) < 0.01
        print('Using default parameters, data is %s, beta is %s' % (graph, beta))
    return (graph, rf_flag, beta, high_order_flag, parameter_flag, pa, pb)


def set_hyperparameter():
    hyperparameter_flag = 'no'
    norm_flag = 'no'
    loss_type = 'hinge'
    n_runs = 10
    pd_flag = 'True'
    multi_cv_flag = True
    n_jobs = -1
    debug_flag = 'off'
    graph_isomorphisim = 'off'
    edge_fil = 'off'
    dynamic_range_flag = True
    return (hyperparameter_flag, norm_flag, loss_type, n_runs, pd_flag, multi_cv_flag, n_jobs, debug_flag, graph_isomorphisim, edge_fil, dynamic_range_flag)

def node_baseline(graph, Y, graphs_, table, allowed, bin_size = 50, extra_feature = [],
                  norm_flag_='yes', cdf_flag=False, edge_structure_flag = False, high_order_flag=False,
                  coarse_flag=True, edge_feature_flag = False, pro_flag=False, skip_flag=False, print_flag=False,
                  laplacian_flag=False, lap_bandwidth=10, uniform_flag=True, sanity_flag = False):
        # graphs_ = compute_graphs_(table, allowed, graph, norm_flag=norm_flag_)
        assert set(allowed).issubset(set(graphs_[0][0].node[0].keys()))
        if edge_feature_flag:
            distance_feature_orgin = get_dist_distribution(n_jobs=-1, print_flag=print_flag, cdf_flag=cdf_flag)
            btwn_feature_origin = edge_btwn_features(upperbound=0.5, pro_flag=pro_flag, n_jobs=-1, dummy_flag=False, print_flag=print_flag)

        baseline_feature = ['1_0_deg_min', '1_0_deg_max', '1_0_deg_mean', 'deg']
        X_origin = merge_features(graph, graphs_, baseline_feature + extra_feature, bin_size, his_norm_flag=norm_flag_, cdf_flag=cdf_flag, uniform_flag=uniform_flag)


        for ax in [0, 1]:
            if (graph == 'reddit_12K') and (ax == 0): # notice that ax 0 is useless for reddit 12k when using linear svm with c = 1000
                continue
            if high_order_flag == 'True':
                from joblib import delayed, Parallel
                high_order_graphs = Parallel(n_jobs=-1)(delayed(high_order)(graphs_[i][0]) for i in range(len(graphs_)))

                # high_order_graphs = [high_order(graphs_[i][0]) for i in range(len(graphs_))]

                high_order_features_2 = high_order_feature(n, high_order_graphs, axis=ax, order=2)
                high_order_features_3 = high_order_feature(n, high_order_graphs, axis=ax, order=3)
                high_order_features_4 = high_order_feature(n, high_order_graphs, axis=ax, order=4)
                high_order_features = np.concatenate((high_order_features_2, high_order_features_3, high_order_features_4), axis=1)
                X_origin = normalize_(X_origin, axis=ax)
                bl_result = baseline(X_origin, Y, graphs_, allowed, n_splits=10, multi_cv_flag=multi_cv_flag, laplacian_flag=laplacian_flag, lap_band=lap_bandwidth)

                print(X_origin, high_order_features)
                X_origin = np.concatenate((X_origin, high_order_features), axis=1)
                print np.shape(X_origin)
                bl_result = baseline(X_origin, Y, graphs_, allowed, n_splits=10, multi_cv_flag=multi_cv_flag, laplacian_flag=laplacian_flag, lap_band=lap_bandwidth)
                print bl_result

            if not skip_flag:
                print('Baseline Model, normalize %s axis' % ax)
                print('Without distance feature')
                X = normalize_(X_origin, axis=ax)

                if coarse_flag:
                    coarseX = merge_features(graph, coarsed_graphs_, baseline_feature + extra_feature, 30, his_norm_flag=norm_flag_, uniform_flag=uniform_flag)
                    coarseX = normalize_(coarseX, axis=ax)
                    X = np.append(X, coarseX, axis=1)
                if sanity_flag == True:
                    n_graph = len(graphs_)
                    extra_feature_ = np.zeros((n_graph, 2))
                    for i in range(n_graph):
                        if len(graphs_[i]) > 0:
                            extra_feature_[i][0] = len(graphs_[i][0])
                            extra_feature_[i][1] = nx.number_of_edges(graphs_[i][0])
                            # extra_feature_[i][2] = nx.diameter(graphs_[i][0])
                        if len(graphs_[i]) > 1:
                            pass
                            # extra_feature_[i][3] = len(graphs_[i][1])
                            # extra_feature_[i][4] = nx.number_of_edges(graphs_[i][1])
                            # extra_feature_[i][5] = nx.diameter(graphs_[i][1])
                    extra_feature_ = normalize_(extra_feature_, axis=ax)
                    print(extra_feature_)
                    X = np.concatenate((X, extra_feature_), axis=1)

                bl_result = baseline(X, Y, graphs_, allowed, n_splits=10, multi_cv_flag=multi_cv_flag, laplacian_flag=laplacian_flag, lap_band=lap_bandwidth, skip_rf=False)
                print('------------------------------------------')
                lapstr = laplacian_str(laplacian_flag, lap_bandwidth)
                table.add_row(
                    [lapstr + 'Bin: ' + str(bin_size) + ' Node BL(deg) ' + str(ax) + str(' ') + str(norm_flag_) + str(extra_feature), bl_result[0],
                     bl_result[1], bl_result[2] + '/' + bl_result[3], ' time:' + str(bl_result[4]) + str(' ') + bl_result[5]])
                print(table)



            if edge_structure_flag:
                X = normalize_(X_origin, axis=ax)
                print('With Structure feature')
                structure_feature = normalize_(edge_histogram(graphs_, attributes=['deg', '1_0_deg_mean', '1_0_deg_sum']), axis=ax)
                X = np.concatenate((X, structure_feature), axis=1)
                X = normalize_(X, axis=ax);
                bl_result = baseline(X, Y, graphs_, allowed, n_splits=10, multi_cv_flag=multi_cv_flag, laplacian_flag=laplacian_flag, lap_band=lap_bandwidth)
                table.add_row(
                    ['Node BL(deg)+Structure Feature ' + str(ax) + str(' ') + str(norm_flag_) + str(extra_feature),
                     bl_result[0],
                     bl_result[1], bl_result[2] + '/' + bl_result[3],
                     ' time:' + str(bl_result[4]) + str(' ') + bl_result[5]])
                print(table)

            if edge_feature_flag:
                X = normalize_(X_origin, axis=ax)
                print('With Distance feature')
                btwn_feature = normalize_(btwn_feature_origin, axis=ax)
                distance_feature = normalize_(distance_feature_orgin, axis=ax)
                # X = np.concatenate((X, distance_feature, btwn_feature), axis=1)
                X = np.concatenate((X, distance_feature), axis=1)
                X = normalize_(X, axis=ax);
                bl_result = baseline(X, Y, graphs_, allowed, n_splits=10, multi_cv_flag=multi_cv_flag ,laplacian_flag=laplacian_flag, lap_band=lap_bandwidth)
                table.add_row(
                    ['Node BL(deg)+Edge Feature ' + str(ax) + str(' ') + str(norm_flag_) + str(extra_feature), bl_result[0],
                     bl_result[1], bl_result[2] + '/' + bl_result[3],
                     ' time:' + str(bl_result[4]) + str(' ') + bl_result[5]])
                print(table)

        return table

def high_order_feature(n, high_order_graphs, order=2, axis=0):
    import numpy as np
    graphs_tmp = [high_order_graphs[i][order-2] for i in range(len(graphs_))]
    n_bins = 40
    features = np.zeros((n, n_bins * 5))

    for i in range(n):
        g = graphs_tmp[i]
        # g = nx.gnp_random_graph(50, 0.5)
        for v in g.nodes():
            g.node[v]['deg'] = g.degree(v)
        for v in g.nodes():
            attribute_mean(g, v, 'deg')
        feature = np.zeros((1,0))
        for attribute in ['1_0_deg_max', '1_0_deg_min', '1_0_deg_mean', '1_0_deg_std', 'deg']:
            lis = nx.get_node_attributes(g, attribute).values()
            feature_ = hisgram(lis, n_bin=n_bins, his_norm_flag='no', lowerbound=0, upperbound=50)
            feature_ = feature_.reshape(1,n_bins)
            feature = np.concatenate((feature, feature_), axis=1)
        features[i] = feature
    features = remove_zero_col(features)
    print('The shape of deg is', np.shape(features))
    return normalize_(features, axis=axis)



if __name__ == '__main__':
    (graph, rf_flag, _, high_order_flag, parameter_flag, pa, pb) = readargs()
    # graph = 'reddit_12K'; rf_flag='no'; pa = 0; pb = 1
    print('pa and pb', pa, pb)
    print('graph is %s'%graph)

    hyperparameter = set_hyperparameter()
    set_global_variable(graph)
    (hyperparameter_flag, norm_flag, loss_type, n_runs, pd_flag, multi_cv_flag, n_jobs, debug_flag, graph_isomorphisim, edge_fil, dynamic_range_flag) = hyperparameter
    threshold = threshold_data(graph)
    (table, n_row ) = set_table(graph, hyperparameter);
    (data, Y_origin) = load_graph(graph);
    Y = Y_origin
    n = len(Y_origin)

    graphs_backup = compute_graphs_(['deg'], graph_isomorphisim, graph, norm_flag='yes', feature_addition_flag=False)
    # graphs_backup = compute_graphs_(['edge_jaccard_minmax'], graph_isomorphisim, graph, norm_flag='yes', feature_addition_flag=False)
    graphs_ = graphs_backup
    print(graphs_backup[0][0].node[0])
    beta = unit_vector(5,3)
    if False:
        (sub_dgms, super_dgms, dgms, _) = handle_edge_filtration('edge_jaccard_minmax')
        beta_name = 'edge_jaccard_minmax'
        for i in range(len(graphs_)):
            export_dgm(graph, sub_dgms[i], i, filename=beta_name, flag='sub')
            export_dgm(graph, super_dgms[i], i, filename=beta_name, flag='super')
            export_dgm(graph, dgms[i], i, filename=beta_name, flag='ss')
            # print('Finish exporting ...', beta_name)
        print sub_dgms
        sys.exit()

    for false_label_percent in [0.0]:
    # for false_label_percent in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print ('false_label_percent is %s\n\n'%false_label_percent)
        table.add_row([' ']*n_row); print(table)
        Y = change_label(graph, Y_origin, change_flag='no', false_label_percent=false_label_percent)
        print Y
        # (data, Y) = load_graph(graph);
        # Y_0 = change_label(graph, Y, change_flag='no', false_label_percent=0.0)
        # (data, Y) = load_graph(graph);
        # Y_1 = change_label(graph, Y, change_flag='no', false_label_percent=1.0)
        # (Y_0 == Y_1).all()
    ### edge filtration
        for edge_fil in allowed_edge:
            continue
            table = edge_baseline(edge_fil, graph, table)
        # baseline and stoa. Compute graphs_ once and use it for multiple times.
        if graph == 'imdb_binary' or graph == 'imdb_multi' or graph =='dd_test' or graph == 'protein_data':
            norm_flag='no'
        else:
            norm_flag='yes'
        graphs_ = graphs_backup
        # graphs_ = compute_graphs_(allowed, graph_isomorphisim, graph, norm_flag='yes', feature_addition_flag=False);
        table = print_stoa(table, graph, n_row)
        for bin_size in [50, 70]:
            continue
            # table = node_baseline(graph, Y, graphs_, table, allowed, bin_size = bin_size, cdf_flag=True, edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False, edge_feature_flag=False, pro_flag=False, high_order_flag='False', laplacian_flag=False, lap_bandwidth=10)
            # for lap_bandwidth in [0.1, 1, 5, 10, 100]:
            table = node_baseline(graph, Y, graphs_, table, allowed, bin_size=bin_size, cdf_flag=True,
                              edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False,
                              edge_feature_flag=False, pro_flag=False, high_order_flag='False', uniform_flag=True, sanity_flag=True)

            table = node_baseline(graph, Y, graphs_, table, allowed, bin_size=bin_size, cdf_flag=True,
                                  edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False,
                                  edge_feature_flag=False, pro_flag=False, high_order_flag='False', uniform_flag=True,
                                  sanity_flag=False)

            table = node_baseline(graph, Y, graphs_, table, allowed, bin_size=bin_size, cdf_flag=False,
                                  edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False,
                                  edge_feature_flag=False, pro_flag=False, high_order_flag='False', uniform_flag=True,
                                  sanity_flag=True)

            table = node_baseline(graph, Y, graphs_, table, allowed, bin_size=bin_size, cdf_flag=False,
                                  edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False,
                                  edge_feature_flag=False, pro_flag=False, high_order_flag='False', uniform_flag=True,
                                  sanity_flag=False)

            # table = node_baseline(graph, Y, graphs_, table, allowed, bin_size=bin_size, cdf_flag=False,
            #                       edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False,
            #                       edge_feature_flag=False, pro_flag=False, high_order_flag='False', uniform_flag=True, sanity_flag=True)
        # table = node_baseline(graph, Y, graphs_, table, allowed, bin_size = bin_size, cdf_flag=False, edge_structure_flag=False, extra_feature=[], norm_flag_='yes', coarse_flag=False, edge_feature_flag=False, pro_flag=False, high_order_flag='False', laplacian_flag=False, lap_bandwidth=10)

    # node filtratillon
    norm_flag = 'yes'
    graphs_ = graphs_backup
    landscape_data = []; itr = 0; svm_data_hist = {0: {}}
    betalist = set_betalist(allowed)
    check_global_safety()
    for beta in betalist:
        assert len(betalist)==3
        print(beta)
        try:
            (beta_name, hop_flag, basep) = set_node_filtration_param(beta, allowed)
        except beta_name_not_in_allowed:
            continue
        (graphs, dgms, sub_dgms, super_dgms, epd_dgms) = dgms_data(graph, beta, n_jobs, debug_flag, hop_flag=hop_flag, basep=basep, edge_fil='off')

        for i in range(len(graphs)):
            continue
            export_dgm(graph, sub_dgms[i], i, filename=beta_name, flag='sub')
            export_dgm(graph, super_dgms[i], i, filename=beta_name, flag='super')
            export_dgm(graph, dgms[i], i, filename=beta_name, flag='ss')
        print('Finish exporting...')

        if True:
            best_vec_result = 0
            for ax in [0, 1]:
                continue
                for epd_flag in [False]:
                    pd_vector_data = clf_pdvector(best_vec_result, (sub_dgms, super_dgms, dgms, epd_dgms), beta, Y, epd_flag=epd_flag, pd_flag='True', print_flag='off', nonlinear_flag=True, axis=ax, rf_flag=rf_flag, dynamic_range_flag=dynamic_range_flag)  # use pd vector as baseline
                    print ('pd_vector_data is ', pd_vector_data)
                    table = add_row(table, pd_vector_data, beta_name, ax, filtration_type='node_vec')
                    print table

                    continue
                    for pvector in [ 'pi', 'pl']:
                        # pd_vector_data = clf_pdvector(best_vec_result, (sub_dgms, super_dgms, dgms, epd_dgms), beta, Y, epd_flag=epd_flag, pvec_flag=True, vec_type=pvector, pd_flag='True', multi_cv_flag=False, print_flag='off', nonlinear_flag=True, axis=ax, rf_flag=rf_flag, dynamic_range_flag=True)
                        # table = add_row(table, pd_vector_data, pvector, ax, filtration_type='node_vec')
                        pd_vector_data = clf_pdvector(best_vec_result, (fake_dgms, fake_dgms, fake_dgms, fake_dgms), beta, Y,
                                                      epd_flag=epd_flag, pvec_flag=True, vec_type=pvector,
                                                      pd_flag='True', multi_cv_flag=False, print_flag='off',
                                                      nonlinear_flag=True, axis=ax, rf_flag=rf_flag,
                                                      dynamic_range_flag=True)
                        table = add_row(table, pd_vector_data, pvector, ax, filtration_type='node_vec')

                        print(table)

        for kernel_type in ['sw']:
            best_result_so_far = (0, 0, {})
            for bandwidth in kernel_parameter(kernel_type)['bw']:
                for k in kernel_parameter(kernel_type)['K']:
                    for p in kernel_parameter(kernel_type)['p']:
                        (true_kernel, _) = (tda_kernel, t1) = sw_parallel(dgms2swdgm(dgms), dgms2swdgm(dgms), parallel_flag=True,
                                                           kernel_type=kernel_type, n_directions=10,
                                                           bandwidth=bandwidth, K=k, p=p)
                        tda_kernel_data_ = evaluate_tda_kernel(tda_kernel, Y, best_result_so_far)
                        if tda_kernel_data_[0] > best_result_so_far[0]:
                            best_result_so_far = tda_kernel_data_  # only print best one
                            table.add_row([beta_name + ' ' + str(kernel_type) + ' bw:' + str(bandwidth), '', '', str(best_result_so_far[0]) + '/' + str(best_result_so_far[1]), 'kct: ' + str(t1) + '/svm_time: ' + str(best_result_so_far[3]) + str(best_result_so_far[2])])
                            print (table)

        if beta[1]==1:
            X = pairing_feature(dgms, n_bin=30, range_=(-1,1))
        else:
            X = pairing_feature(dgms, n_bin=20, range_=(0,1))
        param = searchclf(X, Y, 1002, test_size=0.1)
        evaluate_clf(X, Y, param, n_splits=10)

        X = aggregation_feature(dgms)
        print X
        param = searchclf(X, Y, 1002, test_size=0.1)
        evaluate_clf(X, Y, param, n_splits=10)
        continue

        table = add_row(table, 'dummy', 'dummy', 'dummy', filtration_type='empty')
        print('check graphs_'),
        print(graphs_[0][0].node[0])
        test_norm(super_dgms)
        (matrix_prl, flag3) = load_data(graph, 'matrix_prl', beta)  # load existing matrix
        if flag3 != 'success':
            matrix_prl = Parallel(n_jobs=n_jobs)(
                delayed(get_matrix_i)(i, debug=debug_flag) for i in range(len(dgms)))  # compute matrix in parallel
            dump_data(graph, matrix_prl, 'matrix_prl', beta)
        print('Finish matrix_prl')

        (dist_matrix, idx_dict) = format_matrixprl(matrix_prl)

        if False:
            diags = [dgm2diag_(i) for i in dgms]
            for sigma in [0.1, 1, 10, 100][1:2]:
                dist_matrix = get_roland_matrix_prl(diags, sigma)
                for c in [0.01, 0.1, 1, 10, 100]:
                    kernelsvm(dist_matrix, Y, c, sigma, dist_flag='no')
                    # (kernel, svm_data) = kernelsvm(dist_matrix, Y, c, sigma, dist_flag='no');
        # continue
        c = 0.1; sigma = 0.1
        (kernel, svm_data) = kernelsvm(dist_matrix, Y, c, sigma, dist_flag='yes')
        title_cache = (beta, graph, round, svm_data[0]['train_acc'], svm_data[0]['test_acc'], c, sigma)
        # MDS(dist_matrix, Y, title_cache, gd='True', print_flag='True')
        # continue
        svm_data_hist[itr] = svm_data
        gddata = get_gddata(svm_data)  # yyhat is a dict. yyhat['y'] is a 10-list of numpy array
        cache = (kernel, gddata['y'][0], gddata['y_hat'][0], beta,
                 gddata['alpha'][0])  # [0] here is the index of training number
        if itr > 1:
            check_cache = (svm_data_hist[itr - 1], svm_data_hist[itr]);
            check_trainloss(check_cache)

        title_cache = (beta, graph, round, svm_data[0]['train_acc'], svm_data[0]['test_acc'], c, sigma)
        # MDS(dist_matrix, Y, title_cache, gd='True', print_flag='True')
        train_idx = svm_data[0]['train_idx']
        gradient_ = total_gradient(cache, train_idx, sigma)
        gradient_[0] = 0;
        gradient_[2] = 0;
        gradient_[4] = 0
        total_loss(itr, cache, train_idx)
        landscape_data += [{'beta': beta, 'others': svm_data, 'pd_vector_data': pd_vector_data}]
        write_landscapedata(graph, {'beta': beta, 'others': svm_data})

        beta = beta_update(beta, gradient_)
        beta[0] = 0;
        beta[2] = 0;
        beta[4] = 0;
        beta = beta / float(sum(beta))
        continue

        print('----------------------------------------------------------\n')
        best8 = svm_hyperparameter(dist_matrix, Y, hyperparameter_flag='no')
        title_cache = (beta, graph, round, best8[0][2], best8[0][3], best8[0][0], best8[0][1])
        MDS(dist_matrix, Y, title_cache)

    sys.exit()








