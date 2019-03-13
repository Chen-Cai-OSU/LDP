import networkx as nx
import sys, os
import argparse
import numpy as np
import pickle
import time

from graph import load_graph, function_basis, convert2nx, get_subgraphs, new_norm, save_graphs_
from tunning import hisgram_single_feature, merge_features
from classifier import evaluate_clf, searchclf
from graph import make_direct
from sklearn.preprocessing import normalize
from hyperparameter import load_best_params_

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='reddit_12K')
parser.add_argument('--n_bin', type=int, default=50)
parser.add_argument('--norm_flag', type=str, default='yes')

# for fine tunning
parser.add_argument('--nonlinear_flag', type=str, default='False',
                    help='True means turn on nonlinear kernel for SVM. In most dataset, linear kernel is already good enough. ')
parser.add_argument('--uniform_flag', type=bool, default=True)

if __name__ == '__main__':
    randomseed = 42
    args = parser.parse_args()
    dataset = args.dataset
    bl_feat = ['1_0_deg_min', '1_0_deg_max', '1_0_deg_mean', '1_0_deg_std', 'deg']

    # hyperparameters
    n_bin = args.n_bin # number of bins for historgram
    norm_flag = args.norm_flag  # normalize before calling function_basis versus normalize after
    nonlinear_kernel = args.nonlinear_flag # linear kernel versus nonlinear kernel

    # less important hyperparameter. Used for fine tunning
    his_norm_flag = 'yes'
    uniform_flag = args.uniform_flag # unform versus log scale. True for imdb, False for reddit.
    cdf_flag = True # cdf versus pdf. True for most dataset.


    graphs, labels = load_graph(dataset)
    n = len(graphs)
    graphs_ = []
    direct = os.path.join('../data/cache/', dataset, 'norm_flag_' + str(norm_flag), '')

    try:
        with open(direct + 'graphs_', 'rb') as f:
            t0 = time.time()
            graphs_ = pickle.load(f)
            print('Finish loading existing graphs. Takes %s'%(time.time() - t0))
    except IOError:
        for i in range(n):
            if i % 50 ==0: print('#'),
            gi = convert2nx(graphs[i], i)
            subgraphs = get_subgraphs(gi)
            gi_s = [function_basis(gi, ['deg'], norm_flag=norm_flag) for gi in subgraphs]
            gi_s = [g for g in gi_s if g != None]
            graphs_.append(gi_s)
        if norm_flag == 'no': graphs_ = new_norm(graphs_, bl_feat)
        save_graphs_(graphs_, dataset=dataset, norm_flag=norm_flag)

    print(len(graphs_))

    x_original = merge_features(dataset, graphs_, bl_feat, n_bin, his_norm_flag=his_norm_flag, cdf_flag=cdf_flag, uniform_flag=uniform_flag)
    if norm_flag=='yes':
        x = normalize(x_original, axis = 1)
    else:
        x = x_original
    y = np.array(labels)

    # you may run searchclf to search the hyperparameter yourself, or load the best_hyperparameter alreay computed for you by me
    # best_params_ = searchclf(x, y, randomseed, test_size=0.1, nonlinear_flag=nonlinear_kernel, verbose=0, print_flag='on')
    best_params_ = load_best_params_(dataset)
    print best_params_
    evaluate_clf(x, y, best_params_, 10, n_eval=10)

