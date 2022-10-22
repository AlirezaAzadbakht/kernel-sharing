import torch
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, model, optimizer, scheduler, GPU=4):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.GPU = GPU
        if self.GPU > 0:
            self.multi_model = torch.nn.DataParallel(
                self.model, device_ids=[i for i in range(self.GPU)])

    def train(self, data):
        self.model.train()
        return self.iteration(data, True)

    def test(self, data):
        #        self.model.train()
        #        _ = self.iteration(data, True, True)
        self.model.eval()
        return self.iteration(data, False)

    def iteration(self, data, train=True, only_update_batch=False):
        if train:
            MA_acc = 0
            MA_loss = 6.3
        else:
            tot_acc = []
        for step, (data, label) in enumerate(data):
            if step % 10 == 0:
                print(step)
            if train:
                if self.GPU > 0:
                    logit = self.multi_model(data)
                else:
                    logit = self.model(data)
                pred = torch.argmax(logit, dim=1, keepdim=False)

                loss = F.cross_entropy(logit, label)
                self.optimizer.zero_grad()
                loss.backward()
                if not only_update_batch:
                    self.optimizer.step()
                    loss_float = float(torch.sum(loss.cpu()))
                    acc_float = float(
                        sum((pred == label).float()) / pred.size(0))
                    MA_loss *= 0.9
                    MA_loss += 0.1 * loss_float
                    MA_acc *= 0.9
                    MA_acc += 0.1 * acc_float
                else:
                    if step > 20:
                        break
            else:
                with torch.no_grad():
                    if self.GPU > 0:
                        logit = self.multi_model(data)
                    else:
                        logit = self.model(data)
                    pred = torch.argmax(logit, dim=1, keepdim=False)
                    acc_float = float(sum((pred == label).float()))
                    tot_acc.append(acc_float)
        if not train:
            return sum(tot_acc) / 10000
        return MA_loss, MA_acc
