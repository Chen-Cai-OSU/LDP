#!/usr/bin/env bash
~/anaconda2/bin/python main.py --dataset 'imdb_multi'  --n_bin 100 --norm_flag 'no'
~/anaconda2/bin/python main.py --dataset 'imdb_binary'  --n_bin 70 --norm_flag 'no'

#~/anaconda2/bin/python main.py --dataset 'reddit_binary'  --n_bin 100 --norm_flag 'yes'

exit
for dataset in  'reddit_12K' #'collab' 'reddit_5K' 'reddit_12K'
do
    for n_bin in 30 #30 #50 70 100
    do
        time ~/anaconda2/bin/python main.py --dataset $dataset  --n_bin $n_bin --norm_flag 'yes'
        time ~/anaconda2/bin/python main.py --dataset $dataset  --n_bin $n_bin --norm_flag 'no'
    done
done
