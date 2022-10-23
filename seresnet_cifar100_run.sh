time=$(date '+%d_%m_%Y_%H:%M:%S')
dataset='cifar100'
experiment='shared-cifar100-no-cutout-64-128-256-second-third-stages'
model=se_resnet_d_2_full_shared

CUDA_VISIBLE_DEVICES=1 python ./SEResNet/Cifar.py \
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