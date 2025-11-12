# Run for main results
### CIFAR-N
python LaT_VIB.py --noise_type aggre --dataset cifar10 --sel_ratio 0.4 --pretrain_ep 20 --injection_ep 80  --noise_mode cifarn --num_epochs 200 --is_human

python LaT_VIB.py --noise_type rand1 --dataset cifar10 --sel_ratio 0.4 --pretrain_ep 20 --injection_ep 80  --noise_mode cifarn --num_epochs 200 --is_human

python LaT_VIB.py --noise_type rand2 --dataset cifar10 --sel_ratio 0.4 --pretrain_ep 20 --injection_ep 80  --noise_mode cifarn --num_epochs 200 --is_human

python LaT_VIB.py --noise_type rand3 --dataset cifar10 --sel_ratio 0.4 --pretrain_ep 20 --injection_ep 80  --noise_mode cifarn --num_epochs 200 --is_human

python LaT_VIB.py --noise_type worst --dataset cifar10 --sel_ratio 0.4 --pretrain_ep 20 --injection_ep 80  --noise_mode cifarn --num_epochs 200 --is_human --ub 0.9

python LaT_VIB.py --noise_type noisy100 --dataset cifar100 --sel_ratio 0.4 --pretrain_ep 30 --injection_ep 70  --noise_mode cifarn --num_epochs 200 --is_human --ub 0.9 --z_dim 64

### CIFAR-Sym/Asym
python LaT_VIB.py --dataset cifar10 --sel_ratio 0.4 --pretrain_ep 20 --injection_ep 80  --noise_mode sym --noise_rate 0.2 --num_epochs 200

python LaT_VIB.py --dataset cifar10 --sel_ratio 0.4 --pretrain_ep 20 --injection_ep 80  --noise_mode asym --noise_rate 0.4 --num_epochs 200

python LaT_VIB.py --dataset cifar100 --sel_ratio 0.4 --pretrain_ep 30 --injection_ep 70  --noise_mode sym --noise_rate 0.5 --num_epochs 200 --z_dim 64 --gamma 0.001

### Animal-10N
python LaT_VIB.py --dataset animal-10N --sel_ratio 0.4 --pretrain_ep 20 --injection_ep 80 --num_epochs 200 --noise_mode asym --batch_size 128 

# Run for main ablation
### w/o KI
python LaT_VIB.py --noise_type worst --dataset cifar10 --sel_ratio 0.4 --pretrain_ep 30 --injection_ep 0  --noise_mode cifarn --num_epochs 130 --is_human  --ub 0.9 --thod 0.85 --seed "$i" 
### w/o RT
python LaT_VIB.py --noise_type worst --dataset cifar10 --sel_ratio 0.4 --pretrain_ep 20 --injection_ep 80  --noise_mode cifarn --num_epochs 100 --is_human --ub 0.9 --thod 0.85 --seed "$i"

# Run for $\beta, \gamma$
### $\beta$
python LaT_VIB-Copy1.py --noise_type worst --dataset cifar10 --sel_ratio 0.4 --pretrain_ep 30 --injection_ep 70  --noise_mode cifarn --num_epochs 200 --is_human --ub 0.9 --thod 0.85 --seed "$i" --beta $beta$
### $\gamma$
python LaT_VIB-Copy1.py --noise_type worst --dataset cifar10 --sel_ratio 0.4 --pretrain_ep 30 --injection_ep 70  --noise_mode cifarn --num_epochs 200 --is_human --ub 0.9 --thod 0.85 --seed "$i" --gamma $gamma$
