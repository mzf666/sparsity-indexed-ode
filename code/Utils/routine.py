import torch
from torch.optim.lr_scheduler import _LRScheduler
import warnings
import math
import torch.nn as nn


def accuracy(output, target):
    _, pred = output.topk(5, dim=1)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    correct1 = correct[:, :1].sum().item()
    correct5 = correct[:, :5].sum().item()

    return correct1, correct5


def evaluate(model, criterion, dataloader, device, verbose=True):
    return evaluate_classification(model, criterion, dataloader, device, verbose)


def evaluate_classification(model, criterion, dataloader, device, verbose=True):
    import time
    start = time.time()

    model.eval()
    total = 0
    correct1, correct5 = 0, 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            if not isinstance(output, torch.Tensor):
                output = output.logits

            total += criterion(output, target).item() * data.shape[0]
            correct1_tmp, correct5_tmp = accuracy(output, target)
            correct1 += correct1_tmp
            correct5 += correct5_tmp

    avg_loss = total / len(dataloader.dataset)
    acc1 = 100. * correct1 / len(dataloader.dataset)
    acc5 = 100. * correct5 / len(dataloader.dataset)

    if verbose:
        print(f'Eval: Avg loss: {avg_loss:.4f},'
              f' Top 1 Acc: {correct1}/{len(dataloader.dataset)} ({acc1:.2f}%),'
              f' Top 5 Acc: {correct5}/{len(dataloader.dataset)} ({acc5:.2f}%),'
              f' time: {time.time() - start:.3f}s.\n')

    # model.train()

    return avg_loss, acc1, acc5


def train(model, optimizer, dataloader, device, epoch, verbose=True, log_interval=200):
    model.train()
    model.to(device)
    total_loss = 0.
    correct1, correct5 = 0, 0
    N = 0
    loss_func = nn.CrossEntropyLoss()
    print('\nEpoch {} with training lr {}.'.format(epoch + 1, optimizer.param_groups[0]['lr']))
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if not isinstance(output, torch.Tensor):
            output = output.logits
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.shape[0]
        N += data.size(0)
        correct1_tmp, correct5_tmp = accuracy(output, target)
        correct1 += correct1_tmp
        correct5 += correct5_tmp

        if verbose & ((batch_idx + 1) % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)], Avg loss: {:.6f},'
                  ' Top1-acc: {:2.2f}%, Top5-acc: {:2.2f}%'.format(
                epoch + 1, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), total_loss / N,
                100. * correct1 / N, 100. * correct5 / N,
            ))

    avg_loss = total_loss / len(dataloader.dataset)
    acc1 = 100. * correct1 / len(dataloader.dataset)
    acc5 = 100. * correct5 / len(dataloader.dataset)

    return avg_loss, acc1, acc5


def train_image(model, optimizer, dataloader, device, epoch, verbose=True, log_interval=200):
    model.train()
    model.to(device)
    total_loss = 0.
    correct1, correct5 = 0, 0
    N = 0

    class LabelSmoothing(nn.Module):
        """
        NLL loss with label smoothing.
        """

        def __init__(self, smoothing=0.1):
            """
            Constructor for the LabelSmoothing module.
            :param smoothing: label smoothing factor
            """
            super(LabelSmoothing, self).__init__()
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing

        def forward(self, x, target):
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()

    loss_func = LabelSmoothing()
    print('\nEpoch {} with training lr {}.'.format(epoch + 1, optimizer.param_groups[0]['lr']))
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if not isinstance(output, torch.Tensor):
            output = output.logits
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.shape[0]
        N += data.size(0)
        correct1_tmp, correct5_tmp = accuracy(output, target)
        correct1 += correct1_tmp
        correct5 += correct5_tmp

        if verbose & ((batch_idx + 1) % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)], Avg loss: {:.6f},'
                  ' Top1-acc: {:2.2f}%, Top5-acc: {:2.2f}%'.format(
                epoch + 1, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), total_loss / N,
                100. * correct1 / N, 100. * correct5 / N,
            ))

    avg_loss = total_loss / len(dataloader.dataset)
    acc1 = 100. * correct1 / len(dataloader.dataset)
    acc5 = 100. * correct5 / len(dataloader.dataset)

    return avg_loss, acc1, acc5


def prune(model, pruner, criterion, pruneloader, sparsity, device, train=True, testloader=None):
    model.to(device)
    if train:
        model.train()
        print('\n\tWarning: pruning at model.train() mode ...\n')
    else:
        model.eval()
        print('\n\tWarning: pruning at model.eval() mode ...\n')

    pruner.score(criterion, pruneloader, device, testloader=testloader)
    pruner.mask(sparsity)


def save_dict(save_path, epoch, sparsity, model, optimizer, scheduler, val_acc1, val_acc5, args):
    ckpt = dict(
        val_acc={'acc1': val_acc1, 'acc5': val_acc5},
        epoch=epoch,
        sparsity=sparsity,
        model=model.state_dict(),
        optimizer=optimizer.state_dict() if optimizer is not None else None,
        scheduler=scheduler.state_dict() if scheduler is not None else None,
        args=args,
    )
    torch.save(ckpt, save_path)
    print(f'\nEpoch {epoch} checkpoint saved:\t'
          f'Sparsity = {sparsity * 100: .3f}%,\t'
          f'Acc1 = {val_acc1},\t'
          f'Acc5 = {val_acc5}.\n')


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_warmup, T_max, eta_max=0.1, eta_min=0, last_epoch=-1, verbose=False):
        self.T_warmup = T_warmup
        self.T_cosine = T_max - T_warmup
        self.eta_max = eta_max
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.T_warmup:
            return self._get_lr_warmup()
        else:
            return self._get_lr_cosine()

    def _get_lr_warmup(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.T_warmup == 0:
            return self.base_lrs
        else:
            return [base_lr + (self.eta_max - base_lr) * ((self.last_epoch + 1) / self.T_warmup)
                    for base_lr in self.base_lrs]

    def _get_lr_cosine(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch - self.T_warmup) == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif (self._step_count - self.T_warmup) == 1 and (self.last_epoch - self.T_warmup) > 0:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos((self.last_epoch - self.T_warmup) * math.pi / self.T_cosine)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - self.T_warmup - 1 - self.T_cosine) % (2 * self.T_cosine) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_cosine)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / self.T_cosine)) /
                (1 + math.cos(math.pi * (self.last_epoch - self.T_warmup - 1) / self.T_cosine)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]
