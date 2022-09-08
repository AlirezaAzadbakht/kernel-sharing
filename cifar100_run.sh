time=$(date '+%d_%m_%Y_%H:%M:%S')
dataset='Cifar100'
root='/home/kheradpishehs/azadbakht'
model=convmixer_256_16_shared

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 47760 ./ConvMixer/main.py \
--model $model --drop_path 0.1 --batch_size 128 --lr 1e-2 \
--update_freq 1 --input_size 32 --nb_classes 100 \
--data_path $root/dataset/$dataset/ \
--output $root/dataset/outputs/output_${model}_${dataset}_${time} \
--data_set CIFAR100 \
--epochs 200 \
--use_amp true \
--warmup_epochs 0 \
--weight_decay 0.01 \
--opt_eps 1e-3 \
--clip_grad 1.0 \
--aa rand-m9-mstd0.5-inc1 \
--cutmix 0.5 \
--mixup 0.5 \
--reprob 0.25 \
--remode pixel \
--seed 76 \
#--model_ema true --model_ema_eval true \
                       