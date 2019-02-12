import networkx as nx
from joblib import Parallel, delayed
import pickle
import os, sys
import time
import numpy as np

from tmp import make_direct

def load_graph(graph, debug='off', single_graph_flag=True):
    # exptect label to be numpy.ndarry of shape (n,). However protein_data is different so have to handle it differently
    assert type(graph) == str
    GRAPH_TYPE = graph
    directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/LearningFiltration'
    make_direct(directory)
    inputFile = directory + '/graph+label'
    if os.path.isfile(inputFile):
        start = time.time()
        print('Loading existing files')
        fd = open(inputFile, 'rb')
        if GRAPH_TYPE == 'reddit_12K':
            file = '/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/' + 'reddit_12K' + '.graph'
            f = open(file, 'r')
            data = pickle.load(f)
            graphs_ = data['graph']
            labels_ = data['labels']
        else:
            (graphs_, labels_) = pickle.load(fd)
        print('Loading takes %s' % (time.time() - start))

        if graph == 'ptc':
            graphs_[151] = graphs_[152]
        return (graphs_, labels_)

    print('Start Loading from dataset')
    file = "/Users/admin/Documents/osu/Research/DeepGraphKernels/datasets/dataset/" + GRAPH_TYPE + ".graph"
    if not os.path.isfile(file): file = '/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/' + GRAPH_TYPE + '.graph'
    f = open(file, 'r')
    data = pickle.load(f)
    if single_graph_flag == True: sys.path.append('/home/cai.507/Documents/DeepLearning/GraphSAGE')

    graphs = data['graph']

    labels = data['labels']
    if graph == 'protein_data':
        labels = np.array([-1] * 663 + [1] * 450)
    elif graph == ('nci1' or 'nci109'):
        labels = np.sign(labels - 0.5)
    elif graph == 'ptc':
        graphs[151] = graphs[152]

    if debug == 'on':
        print(graph),
        print(type(labels), )
        print(np.shape(labels))
    print('Finish Loading graphs')
    outputFile = directory + '/graph+label'
    fw = open(outputFile, 'wb')
    dataset = (graphs, labels)
    pickle.dump(dataset, fw)
    fw.close()
    print('Finish Saving data for future use')
    return graphs, labels

def convert2nx(graph, i, print_flag='False'):
    # graph: python dict
    keys = graph.keys()
    try:
        assert keys == range(len(graph.keys()))
    except AssertionError:
        print('%s graph has non consecutive keys'%i)
        print('Missing nodes are the follwing:')
        for i in range(max(graph.keys())):
            if i not in graph.keys(): print i,

    # add nodes
    gi = nx.Graph()
    for i in keys: gi.add_node(i) # change from 1 to i. Something wired here
    assert len(gi) == len(keys)

    # add edges
    for i in keys:
        for j in graph[i]['neighbors']:
            if j > i:
                gi.add_edge(i, j)
    for i in keys:
        # print graph[i]['label']
        if graph[i]['label']=='':
            gi.node[i]['label'] = 1
            # continue
        try:
            gi.node[i]['label'] = graph[i]['label'][0]
        except TypeError: # modifications for reddit_binary
            gi.node[i]['label'] = graph[i]['label']
        except IndexError:
            gi.node[i]['label'] = 0 # modification for imdb_binary
    assert len(gi.node) == len(graph.keys())
    gi.remove_edges_from(gi.selfloop_edges())
    if print_flag=='True': print('graph: %s, n_nodes: %s, n_edges: %s' %(i, len(gi), len(gi.edges)) )
    return gi


def attribute_mean(g, i, key='deg', cutoff=1, iteration=0):
    # g = graphs_[i][0]
    # g = graphs_[0][0]
    # attribute_mean(g, 0, iteration=1)
    for itr in [iteration]:
        assert key in g.node[i].keys()
        # nodes_b = nx.single_source_shortest_path_length(g,i,cutoff=cutoff).keys()
        # nodes_a = nx.single_source_shortest_path_length(g,i,cutoff=cutoff-1).keys()
        # nodes = [k for k in nodes_b if k not in nodes_a]
        nodes = g[i].keys()

        if iteration == 0:
            nbrs_deg = [g.node[j][key] for j in nodes]
        else:
            key_ = str(cutoff) + '_' + str(itr-1) + '_' + key +  '_' + 'mean'
            nbrs_deg = [g.node[j][key_] for j in nodes]
            g.node[i][ str(cutoff) + '_' + str(itr) + '_' + key] = np.mean(nbrs_deg)
            return

        oldkey = key
        key = str(cutoff) + '_' + str(itr) + '_' + oldkey
        key_mean = key + '_mean'; key_min = key + '_min'; key_max = key + '_max'; key_std = key + '_std'
        key_sum = key + '_sum'

        if len(nbrs_deg) == 0:
            g.node[i][key_mean] = 0
            g.node[i][key_min] = 0
            g.node[i][key_max] = 0
            g.node[i][key_std] = 0
            g.node[i][key_sum] = 0
        else:
            # assert np.max(nbrs_deg) < 1.1
            g.node[i][key_mean] = np.mean(nbrs_deg)
            g.node[i][key_min] = np.min(nbrs_deg)
            g.node[i][key_max] = np.max(nbrs_deg)
            g.node[i][key_std] = np.std(nbrs_deg)
            g.node[i][key_sum] = np.sum(nbrs_deg)

def function_basis(g, allowed, norm_flag = 'no'):
    # input: g
    # output: g with ricci, deg, hop, cc, fiedler computed
    # allowed = ['ricci', 'deg', 'hop', 'cc', 'fiedler']
    # to save recomputation. Look at the existing feature at first and then simply compute the new one.

    if len(g)<3: return
    assert nx.is_connected(g)

    def norm(g, key, flag=norm_flag):
        if flag=='no':
            return 1
        elif flag == 'yes':
            return np.max(np.abs(nx.get_node_attributes(g, key).values())) + 1e-6

    if 'deg' in allowed:
        deg_dict = dict(nx.degree(g))
        for n in g.nodes():
            g.node[n]['deg'] = deg_dict[n]
            # g_ricci.node[n]['deg'] = np.log(deg_dict[n]+1)

        deg_norm = norm(g, 'deg', norm_flag)
        for n in g.nodes():
            g.node[n]['deg'] /= np.float(deg_norm)
    if 'deg' in allowed:
        for n in g.nodes():
            attribute_mean(g, n, key='deg', cutoff=1, iteration=0)
        if norm_flag == 'yes':
            # better normalization
            for attr in [ '1_0_deg_sum']: # used to include 1_0_deg_std/ deleted now:
                norm_ = norm(g, attr, norm_flag)
                for n in g.nodes():
                    g.node[n][attr] /= float(norm_)
    return g

def get_subgraphs(g, threshold=1):
    assert str(type(g)) == "<class 'networkx.classes.graph.Graph'>"
    subgraphs = [g.subgraph(c).copy() for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    subgraphs = [c for c in subgraphs if len(c) > threshold]
    return subgraphs

def load_data(GRAPH_TYPE, dataname, beta=-1, no_load='yes'):
    # load data from a directory
    import time
    start = time.time()
    if no_load == 'yes':
        return (None, 'Failure')
    import pickle, os
    directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/Baseline/'
    if (str(type(beta))=="<type 'numpy.ndarray'>") and (len(beta)==5):
        beta_str = str(beta[0]) + '_' + str(beta[1]) + '_' + str(beta[2]) + '_' + str(beta[3]) + '_' + str(beta[4])
        directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/LearningFiltration/' + beta_str + '/'
        make_direct(directory)

    inputFile = directory + dataname
    if os.path.isfile(inputFile):
        import cPickle as pickle
        print('Loading existing files %s'%dataname)
        fd = open(inputFile, 'rb')
        dataset = pickle.load(fd)
        print('Loading %s takes %s'%(dataname, time.time()- start))
        return (dataset, 'success')
    else:
        return (None, 'Failure')
def dump_data(GRAPH_TYPE, dataset, dataname, beta=-1, still_dump='yes', skip='yes'):
    if skip=='yes':
        return
    # save dataset in a directory for future use
    import pickle, os
    directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/Baseline/'; make_direct(directory)
    if (str(type(beta))=="<type 'numpy.ndarray'>") and (len(beta)==5):
        print('Saving in beta subdirectory')
        beta_str = str(beta[0]) + '_' +  str(beta[1]) + '_' + str(beta[2]) + '_' + str(beta[3]) + '_' + str(beta[4])
        directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + GRAPH_TYPE + '/Baseline/' + beta_str + '/'
        make_direct(directory)

    outputFile = directory + dataname
    if os.path.exists(outputFile):
        if still_dump == 'no':
            print('File already exists. No need to dump again.')
            return
    print('Dumping')
    fw = open(outputFile, 'wb')
    pickle.dump(dataset, fw)
    fw.close()
    print('Finish Saving data %s for future use'%dataname)

def compute_graphs_(allowed, graph, norm_flag = 'no', feature_addition_flag=False, skip_dump_flag='yes'):
    (graphs_tmp, message) = load_data(graph, 'dgms_normflag_'+norm_flag, beta=-1, no_load='yes')
    if message=='success':
        return graphs_tmp

    (graphs_tmp, flag) = load_data(graph, 'graphs_', no_load='yes')
    if flag != 'success':
        from joblib import Parallel, delayed
        sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/')
        from cycle_basis_v2 import pipeline0
        graphs_tmp = Parallel(n_jobs=-1, batch_size='auto')(delayed(pipeline0)(i, allowed, norm_flag = norm_flag, feature_addition_flag=feature_addition_flag) for i in range(len(data)))
        dump_data(graph, graphs_tmp, 'graphs_', skip=skip_dump_flag)
    print ''
    dump_data(graph, graphs_tmp, 'dgms_normflag_'+norm_flag, beta=-1, still_dump='yes', skip='yes')

    if (graph =='imdb_binary') or (graph == 'imdb_multi') or (graph == 'dd_test') or (graph=='protein_data') or (graph == 'collab'):
        uniform_norm_flag=True
    else:
        uniform_norm_flag=False

    if norm_flag == 'yes': uniform_norm_flag = False

    if uniform_norm_flag:
        anynode = list(graphs_tmp[0][0].nodes)[0]
        print(graphs_tmp[0][0].nodes[anynode])
        attribute_lists = graphs_tmp[0][0].nodes[anynode].keys()
        attribute_lists = [attribute for attribute in attribute_lists if attribute!='hop']
        for attribute in attribute_lists:
            max_ = 0; tmp_max_ = []
            min_ = 1; tmp_min_ = []
            for i in range(len(graphs_tmp)):
                if len(graphs_tmp[i])==0:
                    print('skip graph %s'%i)
                    continue
                tmp_max_ += [np.max(nx.get_node_attributes(graphs_tmp[i][0], attribute).values())] # catch exception
                # except:
                #     print (i, graphs_tmp[i])
                tmp_min_ += [np.min(nx.get_node_attributes(graphs_tmp[i][0], attribute).values())]
            from heapq import nlargest
            print nlargest(5, tmp_min_)[-1]
            denominator = max(nlargest(10, tmp_max_)[-1],  nlargest(10, np.abs(tmp_min_))[-1]) +1e-10
            # denominator = max(max(np.abs(tmp_max_)),max(np.abs(tmp_min_)))+1e-10
            print ('Attribute and demoninator: ', attribute, denominator)

            for i in range(len(graphs_tmp)):
                if len(graphs_tmp[i]) == 0:
                    continue
                n_component = len(graphs_tmp[i])
                for comp in range(n_component):
                    for v in graphs_tmp[i][comp].nodes():
                        graphs_tmp[i][comp].nodes[v][attribute] = graphs_tmp[i][comp].nodes[v][attribute]/ np.float(denominator)
                        # graphs_tmp[i][comp].nodes[v][attribute + '_uniform'] = graphs_tmp[i][comp].nodes[v][attribute]/ np.float(denominator)
                        # print graphs_tmp[i][0].nodes[v][attribute],

    return graphs_tmp

def new_norm(graphs_, bl_feat):
    """Normalize graph function uniformly"""
    newnorm = dict(zip(bl_feat, [0] * 5))
    for attr in bl_feat:
        for gs in graphs_:
            for g in gs:
                tmp = max(nx.get_node_attributes(g, attr).values())
                if tmp > newnorm[attr]:
                    newnorm[attr] = tmp

    for gs in graphs_:
        for g in gs:
            for n in g.nodes():
                for attr in bl_feat:
                    g.node[n][attr] /= float(newnorm[attr])
                    assert g.node[n][attr] <=1
    return graphs_

def save_graphs_(graphs_, dataset='imdb_binary', norm_flag='yes'):
    t0 = time.time()
    direct = os.path.join('/home/cai.507/Documents/DeepLearning/LDP/data/cache/', dataset, 'norm_flag_' + str(norm_flag), '')
    if not os.path.exists(direct): make_direct(direct)
    with open(direct + 'graphs_', 'wb') as f:
        pickle.dump(graphs_, f)
    print('Saved graphs. Takes %s'%(time.time() - t0))

def load_best_params_(dataset):
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