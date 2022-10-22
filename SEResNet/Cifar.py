import argparse
import time

import random

import torch.optim as optim
from Training import Trainer, DataLoader
import senet

import time
from logger import log_print

def main():
    loader = DataLoader(aug=args.aug, cutout=args.cutout, dataset=args.dataset)
    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    if args.network in dir(senet):
        model = getattr(senet, args.network)(
            num_classes=num_classes, new_resnet=args.new_resnet, dropout=args.dropout, sync=args.sync)
    else:
        raise ValueError('no such model')
    model.cuda()
    print(model)
    input()
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
    #                                         print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # input()
    optimizer = optim.SGD(params=model.parameters(),
                          lr=args.lr,
                          momentum=args.m,
                          weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[85, 130, 180], gamma=0.1)
    trainer = Trainer(model, optimizer, scheduler, args.GPU)
    his_max_acc = []
    log_print(f'{args.network}, Number of Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}', logfile)
    # input()
    for e in range(args.epochs):
        t0 = time.time()
        loss, acc = trainer.train(loader.generator(True,
                                                   args.batch_size,
                                                   args.GPU))
        t1 = time.time()
        log_print(
            f'epoch{e+1}/{args.epochs}|train_acc={acc}|loss={loss}|time_lapse={t1-t0}s', logfile)

        t0 = time.time()
        acc = trainer.test(loader.generator(False,
                                            args.batch_size,
                                            args.GPU))
        his_max_acc.append(acc)
        t1 = time.time()
        log_print(
            f'epoch{e+1}/{args.epochs}|test_acc={acc}|best_acc={max(his_max_acc)}|time_lapse|{t1-t0}s', logfile)
        scheduler.step()
    return his_max_acc


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--experiment_name", type=str, default='basleine')
    parser.add_argument("--network", type=str, default='se_resnet110')
    parser.add_argument("--GPU", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--m", type=float, default=9e-1)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--aug", action='store_true')
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--no-sync', dest='sync', action='store_false')
    parser.set_defaults(sync=True)
    parser.add_argument("--new_resnet", action='store_true')
    parser.add_argument("--cutout", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.,
                        help="probability of discarding features")
    parser.add_argument("--dataset", type=str, default='cifar10')
    args = parser.parse_args()
    timestamp = int(time.time())
    experiment_name = f'{args.experiment_name}-{args.network}'
    logfile = open(f'logs/log_{experiment_name}_{timestamp}.txt', 'w')

    h_acc = main()
    ID = f'{random.random():.6f}'
    log_print(f'saved to: ID = {ID}', logfile)
    with open(f'./result-{ID}.txt', 'w')as f:
        print(','.join(map(str, h_acc)), file=f)
