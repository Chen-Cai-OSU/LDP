# LDP
This is the implementation of paper "A simple yet effective baseline for non-attribute graph classification"

- https://arxiv.org/abs/1811.03508
- In this paper, we showed the existing benchmark dataset for graph classification is too easy to evaluating existing graph classification methods. Many exisiting methods are either based on graph kernel or graph neural networks demonstrated their power on existing dataset, but we found that using very simple feature based on degree statistics + SVM can already yield comparable results on multiple dataset.
- dataset can be downloaded at [here](http://www.mit.edu/~pinary/kdd/)


- make sure networkx>=2.2, joblib>=0.11, sklearn>='0.19.1'
- To find the exact hyperparameters to replicated the results in the paper, you may refer hyperparameter.py
- to Run the code, go to LDP/code/ directory and execute python main.py

