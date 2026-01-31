cd ../

python clustering_llava.py --k_mean 10 --pretrain_path "/home/maitanha/vgu/cll_thuan/balanced_cll_pretrained/CIFAR10/CIFAR10_checkpoint_0799_-0.8583.pth.tar"

python clustering_llava.py --dataset cifar20 --k_mean 20 --pretrain_path "/home/maitanha/vgu/cll_thuan/balanced_cll_pretrained/CIFAR20/CIFAR20_checkpoint_0799_-0.8386.pth.tar"

python clustering_llava.py --dataset cifar100 --k_mean 100 --pretrain_path "/tmp2/maitanha/vgu/cll_thuan/balanced_cll_pretrained/CIFAR100/CIFAR100_checkpoint_0799_-0.8792.pth.tar"

python clustering_llava.py --dataset tiny200 --k_mean 200 --pretrain_path "/tmp2/maitanha/vgu/cll_thuan/balanced_cll_pretrained/Tiny200/TinyImageNet_checkpoint_0799_-0.8987.pth.tar"

python clustering_llava.py --k_mean 10 --decoder simclr --pretrain_path "/tmp2/maitanha/solo-learn/trained_models/simclr/0chuba1b/simclr-cifar10-0chuba1b-ep=799.ckpt"

python clustering_llava.py --k_mean 10 --decoder byol --pretrain_path "/tmp2/maitanha/pretrained model/balanced_cll_pretrained/CIFAR10/byol-cifar10-32brzx9a-ep=999.ckpt"

python clustering_llava.py --k_mean 10 --decoder mocov3 --pretrain_path "/tmp2/maitanha/solo-learn/trained_models/mocov3/nwwlwbq0/mocov3-cifar10-nwwlwbq0-ep=999.ckpt"
