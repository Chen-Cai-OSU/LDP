# LDP
This is the implementation of paper "A simple yet effective baseline for non-attribute graph classification", accepted by ICLR 2019 workshop on Representation learning on graphs and manifolds

*To better understand the machinery of various methods for the graph classification task, we develop a simple yet meaningful graph representation, and explore its effectiveness and limitation.  Interestingly, this degree based simple representation achieves similar performance as the state-of-the-art graph kernels and graph neural networks for non-attributed graph classification. Its connection to graph neural networks and Weisfeiler-Lehman kernel is also presented.*


- https://arxiv.org/abs/1811.03508
- In this paper, we showed the existing benchmark dataset for graph classification is too easy to evaluate existing graph classification methods. Existing methods either based on graph kernel or graph neural networks demonstrated their power on the existing dataset, but we found that using very simple features based on degree statistics + SVM can already yield comparable results on multiple datasets.
- This repo only contains two small datasets (Imdb_binary and Imdb_multi). The full dataset can be downloaded at [here](http://www.mit.edu/~pinary/kdd/)
- make sure networkx>=2.2, joblib>=0.11, sklearn>=0.19.1
- To find the exact hyperparameters to replicated the results in the paper, you may refer hyperparameter.py
- to Run the code, go to LDP/code/ directory and execute run.sh
- If you find our paper interesting, please use the following BibTeX format for citation. 
 
 ```
@article{cai2018simple,
  title={A simple yet effective baseline for non-attribute graph classification},
  author={Cai, Chen and Wang, Yusu},
  journal={arXiv preprint arXiv:1811.03508},
  year={2018}
}
```


