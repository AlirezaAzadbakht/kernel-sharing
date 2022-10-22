time=$(date '+%d_%m_%Y_%H:%M:%S')
dataset='cifar10'
experiment='shared-cifar10-no-cutout-64-128-256-second-third-stages'
model=se_resnet_d_4

CUDA_VISIBLE_DEVICES=1 /home/kherad/anaconda3/envs/alireza/bin/python ./SEResNet/Cifar.py \
--experiment_name $experiment \
--sync \
--epochs 256 \
--GPU 1 \
--batch_size 128 \
--lr 1e-1 \
--m 9e-1 \
--wd 1e-4 \
--cutout 0 \
--dropout 0 \
--dataset $dataset \
--network $model                       