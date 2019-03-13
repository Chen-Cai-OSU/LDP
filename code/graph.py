import networkx as nx
from joblib import Parallel, delayed
import pickle
import os, sys
import time
import numpy as np

def make_direct(direct):
    # has side effect
    import os
    if not os.path.exists(direct):
            os.makedirs(direct)

def load_graph(graph, debug='off', single_graph_flag=True):
    # exptect label to be numpy.ndarry of shape (n,). However protein_data is different so have to handle it differently
    assert type(graph) == str
    directory = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + graph + '/LearningFiltration'
    # make_direct(directory)
    # inputFile = directory + '/graph+label'
    #
    # if os.path.isfile(inputFile):
    #     start = time.time()
    #     print('Loading existing files')
    #     fd = open(inputFile, 'rb')
    #     if graph == 'reddit_12K':
    #         file = '/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/' + 'reddit_12K' + '.graph'
    #         f = open(file, 'r')
    #         data = pickle.load(f)
    #         graphs_ = data['graph']
    #         labels_ = data['labels']
    #     else:
    #         (graphs_, labels_) = pickle.load(fd)
    #     print('Loading takes %s' % (time.time() - start))

    print('Start Loading from dataset')
    file = os.path.join("../data/", graph + ".graph")
    # if not os.path.isfile(file): file = '/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/' + graph + '.graph'
    f = open(file, 'r')
    data = pickle.load(f)
    graphs, labels = data['graph'], data['labels']
    return graphs, labels

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
        # print('%s graph has non consecutive keys'%i)
        # print('Missing nodes are the follwing:')
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
    direct = os.path.join('../data/cache/', dataset, 'norm_flag_' + str(norm_flag), '')
    if not os.path.exists(direct): make_direct(direct)
    with open(direct + 'graphs_', 'wb') as f:
        pickle.dump(graphs_, f)
    print('Saved graphs. Takes %s'%(time.time() - t0))

def stoa():
    mlg = {'mutag': '87.4+/-1.61',  'ptc': '63.26(+/-1.48)' , 'enzyme': '61.81(+/-0.99)',  'protein_data': '76.34(+/-0.72)',  'nci1': '81.75(+/-0.24)', 'nci109': '81.31(+/-0.22)'}
    wl = {'mutag': '84.50(+/-2.16)', 'ptc': '59.97(+/-1.60)', 'enzyme': '53.75(+/-1.37)', 'protein_data': '75.49(+/-0.57)', 'nci1': '84.76(+/-0.32)', 'nci109': '85.12(+/-0.29)'}
    wl_edge = {'mutag': '82.94(+/-2.33)',  'ptc': '60.18(+/-2.19)', 'enzyme': '52.00(+/-0.72)',  'protein_data': '74.78(+/-0.59)', 'nci1': '84.65(+/-0.25)', 'nci109': '85.32(+/-0.34)'}
    fgsd = {'mutag': '92.12', 'ptc': '62.8', 'protein_data': '73.42', 'nci1': '79.8', 'nci109': '78.84', 'dd': '77.10', 'mao': '95.59',
            'reddit_binary': '86.5', 'reddit_5K': '47.76', 'reddit_12K': '47.76', 'imdb_binary': '73.62', 'imdb_multi': '52.41', 'collab': '80.02'}
    roland = {'reddit_5K': '54.5', 'reddit_12K': '44.5'}
    retgk = {'mutag': '90.31', 'ptc': '62.5', 'enzyme': '60.4', 'protein_data': '75.8', 'nci1': '84.5', 'dd': '81.6', 'collab': '81.0', 'imdb_binary': '72.3',
             'imdb_multi': '47.7', 'reddit_binary': '92.6', 'reddit_5K': '56.1', 'reddit_12K': '48.7'}
    deg_baseline={'mutag': '90.07',
                  'ptc': '61.7(50bin)/64.5(+label)/',
                  'protein_data': '71.4/72.5(50bin+cdf, 73.3 if add pair dist)/\n73.7(+label)/74.7(+label + 50bin)',
                  'nci1': '71/74.7(fine tune)',
                  'dd':'75.35/76.2(+ricci+label)/77.5(deg+dist+btwn)/\n77.5(deg+label+ricci+dist+btwn)/77.8(+ new norm)',
                  'enzyme': '35.6(+label)/38.5(+label+ricci)',
                  'reddit_binary': '90.27/91.4(+dist distribution)/\n91.6(+btwn)/92.1(cdf,100bin)',
                  'imdb_binary': '70/72.6(+edge dist and btwness)\n/74.0(+dist and btwn(300bin + 0.5ub))/ 75.4(new norm + cdf)',
                  'imdb_multi':'45(svm)/48(rf)/48.5(rf+btwn, dist feature)\n /49.0(rf+btwn, dist feature fine tunning)\n/50.0(new norm +cdf) /50.8(new norm + cc + edge feature)',
                  'reddit_5K': '53.8/54.0(+deg sum)\n /54.4(cdf + deg_sum)/54.9(log)/\n55.9(log+log 30bin)/log scale + 30 bin 55.9',
                  'reddit_12K':'43/44.0+deg sum/\n nonlinear kernel + log scale 47.8',
                  'collab': '74.7/77.0(rf)/\n77.1(+dist distribution)/\n77.6(rf + old norm +100 bin)\n 78.2(new norm + extra feature + 70 bin + cdf)'}

    return {'mlg': mlg, 'wl': wl, 'wl_edge':wl_edge, 'fgsd': fgsd, 'roland': roland, 'deg_baseline': deg_baseline, 'retgk': retgk}

