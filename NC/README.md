
# Run for main experiment
### Cora
python LaT_GIB.py --noise_ratio $nr --lr 0.005 --warmup 10 --sel_ratio 0.6 --injection 30 --hid_dim 20 --ub 0.9 --thod 0.8 --random_state $seed --dataset Cora (--noise_type pair)
### Citeseer
python LaT_GIB.py --noise_ratio $nr --lr 0.005 --warmup 5 --sel_ratio 0.6 --injection 15 --hid_dim 20 --ub 0.9 --thod 0.9 --random_state $seed --dataset Citeseer (--noise_type pair)
### Pubmed
python LaT_GIB.py --noise_ratio $nr --lr 0.005 --warmup 40 --sel_ratio 0.7 --injection 30 --hid_dim 16 --ub 0.5 --lb 0.3 --thod 0.90 --random_state $seed --dataset Pubmed  (--noise_type pair)

### DBLP (20 sample per class)
python LaT_GIB.py --noise_ratio $nr --lr 0.005 --warmup 10 --sel_ratio 0.6 --injection 30 --hid_dim 16 --ub 0.5 --lb 0.3 --thod 0.7 --random_state $seed --split per_class --dataset dblp (--noise_type pair)

### DBLP (0.01:0.15:0.84)
python LaT_GIB.py --noise_ratio 0.4 --lr 0.005 --warmup 20 --sel_ratio 0.6 --injection 40 --hid_dim 16 --ub 0.6 --lb 0.3 --thod 0.6 --random_state 0 --dataset dblp (--noise_type pair)

# Run for ablation
### w/o KI
python LaT_GIB.py --noise_ratio 0.4 --lr 0.005 --warmup 40 --sel_ratio 0.7 --injection 0 --hid_dim 16 --ub 0.5 --lb 0.3 --thod 0.90 --random_state $seed --dataset Pubmed
### w/o RT
python LaT_GIB.py --noise_ratio 0.4 --lr 0.005 --warmup 40 --sel_ratio 0.7 --injection 60 --hid_dim 16 --ub 0.5 --lb 0.3 --thod 0.90 --random_state $seed --dataset Pubmed
# Run for selection ratio $\delta$
python LaT_GIB.py --noise_ratio $nr --lr 0.005 --warmup 10 --sel_ratio $sel_ratio --injection 30 --hid_dim 20 --ub 0.9 --thod 0.8 --random_state $seed

(for sel_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)