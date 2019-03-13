#!/usr/bin/env bash
# replicate results for imdb_binary and imdb_multi using linear kernel

~/anaconda2/bin/python main.py  --dataset 'imdb_multi'  --n_bin 100 --norm_flag 'no'
~/anaconda2/bin/python main.py  --dataset 'imdb_binary'  --n_bin 70 --norm_flag 'no'
