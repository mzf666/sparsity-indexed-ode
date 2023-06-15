import copy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from Methods.mask_processor import load_mask_procer, reg_mask_proc_hooks
from Methods.utils import masked_params, classification_helper, transformers_helper
from Utils.misc import func_timer
from torch.distributions import Normal


class Pruner:
    def __init__(self, model):
        self.model = model
        self.scores = {}
        for key, m, _ in masked_params(self.model):
            self.scores[key] = torch.ones_like(m).to(m)

    def score(self, criterion, dataloader, device, testloader=None):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        for key, mask, _ in masked_params(self.model):
            self.scores[key] *= mask

        global_scores = torch.cat([s.flatten() for s in self.scores.values()])
        prune_num = int((1 - sparsity) * global_scores.numel())
        if not prune_num < 1:
            threshold, _ = global_scores.kthvalue(k=prune_num)
            for key, mask, _ in masked_params(self.model):
                score = self.scores[key]
                zero = torch.tensor([0.]).to(mask)
                one = torch.tensor([1.]).to(mask)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def _local_mask(self, sparsity):
        for key, mask, _ in masked_params(self.model):
            self.scores[key] *= mask

        for (key, mask, _), s in zip(masked_params(self.model), sparsity):
            score = self.scores[key]
            prune_num = int((1 - s) * score.numel())
            if not prune_num < 1:
                threshold, _ = score.flatten().kthvalue(k=prune_num)
                zero = torch.tensor([0.]).to(mask)
                one = torch.tensor([1.]).to(mask)
                mask.copy_(torch.where(score <= threshold, zero, one))

    @func_timer
    def mask(self, sparsity):
        if isinstance(sparsity, list):
            self._local_mask(sparsity)
        else:
            self._global_mask(sparsity)

    def apply_mask(self):
        for _, m, p in masked_params(self.model):
            p.data *= (m != 0).float()
        print('Model pruned.')

    def stats(self, model):
        sparsities = []
        remaining, total = 0, 0
        # for _, _, p in masked_params(model):
        #     remaining += p.data.count_nonzero().item()
        #     total += p.data.numel()

        for _, m, _ in masked_params(model):
            remaining += m.data.count_nonzero().item()
            total += m.data.numel()
            sparsities += [m.data.count_nonzero().item() / m.data.numel()]

        print('Structural sparsity:', sparsities)

        return remaining, total, remaining / total


class Rand(Pruner):
    def __init__(self, model):
        super(Rand, self).__init__(model)

    @func_timer
    def score(self, criterion, dataloader, device, testloader):
        for key, _, w in masked_params(self.model):
            self.scores[key] = torch.randn_like(w).to(w)


class Mag(Pruner):
    def __init__(self, model):
        super(Mag, self).__init__(model)

    @func_timer
    def score(self, criterion, dataloader, device, testloader):
        for key, _, w in masked_params(self.model):
            self.scores[key] = w.detach().abs()


class MagRand(Pruner):
    def __init__(self, model):
        super(MagRand, self).__init__(model)

    @func_timer
    def score(self, criterion, dataloader, device, testloader):
        for key, _, p in masked_params(self.model):
            self.scores[key] = (p * torch.randn_like(p).to(p)).detach().abs()


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, model):
        super(SNIP, self).__init__(model)

    @func_timer
    def score(self, criterion, dataloader, device, testloader):

        # allow masks to have gradient
        for _, m, _ in masked_params(self.model):
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = self.model(data)
            criterion(output, target).backward()

        # calculate score |g * theta|
        for key, m, p in masked_params(self.model):
            self.scores[key] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for key, _, _ in masked_params(self.model):
            self.scores[key].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, model):
        super(GraSP, self).__init__(model)
        self.temp = 200
        self.eps = 1e-10

    @func_timer
    def score(self, criterion, dataloader, device, testloader):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = self.model(data) / self.temp
            L = criterion(output, target)

            grads = torch.autograd.grad(L, [p for _, _, p in masked_params(self.model)], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = self.model(data) / self.temp
            L = criterion(output, target)

            grads = torch.autograd.grad(L, [p for _, _, p in masked_params(self.model)], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()

        # calculate score Hg * theta (negate to remove top percent)
        for key, _, p in masked_params(self.model):
            self.scores[key] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for key, _, p in masked_params(self.model):
            self.scores[key].div_(norm)


class SynFlow(Pruner):
    def __init__(self, model):
        super(SynFlow, self).__init__(model)

    @func_timer
    def score(self, criterion, dataloader, device, testloader):

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(self.model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        input = torch.ones([1] + input_dim).to(device)  # , dtype=torch.float64).to(device)
        self.model.train()
        output = self.model(input)
        self.model.zero_grad()
        torch.sum(output).backward()

        for key, _, p in masked_params(self.model):
            if torch.isnan(p.grad.data.sum()):
                print(p.grad.data)
            self.scores[key] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(self.model, signs)


class ODE(Pruner):
    def __init__(self, model, N=15, r=1, ode_scope='global', E='CE', G='l1',
                 score_option='m', mask_option='one', mask_proc_kwargs={},
                 schedule='lin', rt_schedule='fix', momentum=0, start=None,
                 save_dir=None, save_ckpt=False):
        super(ODE, self).__init__(model)

        from Methods.sparsity_ode import SparsityIndexedODE
        self.ode = SparsityIndexedODE(None, N, r, ode_scope, E, G,
                                      score_option, mask_proc_kwargs,
                                      schedule, rt_schedule, momentum,
                                      save_dir, save_ckpt)
        self.score_option = score_option
        self.mask_option = mask_option
        self.start = start

    @func_timer
    def score(self, criterion, dataloader, device, testloader):
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.testloader = testloader
        self.ode.discretization = partial(self.ode.discretization, testloader=testloader)

    def _global_mask(self, sparsity):
        for key, _, p in masked_params(self.model):
            self.scores[key] *= (p != 0).float()

        global_scores = torch.cat([s.flatten() for s in self.scores.values()])
        prune_num = int((1 - sparsity) * global_scores.numel())
        if not prune_num < 1:
            threshold, _ = global_scores.kthvalue(k=prune_num)
            for key, mask, _ in masked_params(self.model):
                score = self.scores[key]
                zero = torch.tensor([0.]).to(mask)

                if self.mask_option == 'one':
                    one = torch.tensor([1.]).to(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, one))

                elif self.mask_option == 'mask':
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, mask_copy))

                elif self.mask_option == 'sign':
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, mask_copy.sign()))

    def _local_mask(self, sparsity):
        for key, _, p in masked_params(self.model):
            self.scores[key] *= (p != 0).float()

        for (key, mask, _), s in zip(masked_params(self.model), sparsity):
            score = self.scores[key]
            prune_num = int((1 - s) * score.numel())
            if not prune_num < 1:
                threshold, _ = score.flatten().kthvalue(k=prune_num)
                zero = torch.tensor([0.]).to(mask)

                if self.mask_option == 'one':
                    one = torch.tensor([1.]).to(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, one))

                elif self.mask_option == 'mask':
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, mask_copy))

                elif self.mask_option == 'sign':
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, mask_copy.sign()))

    @func_timer
    def mask(self, sparsity):
        # freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # allow masks to have gradient
        for _, m, _ in masked_params(self.model):
            m.requires_grad = True
            m.grad = None

        # run Sparsity-Indexed ODE
        if self.start is None:
            if isinstance(sparsity, list):
                self.start = [1.] * len(sparsity)
            else:
                self.start = 1.

        self.end = sparsity
        if not hasattr(self, 'quant_path'):
            self.quant_path = None

        self.ode.discretization(self.start, self.end, self.model, self.dataloader, self.device, self.quant_path)
        self.start = sparsity

        # scores <-- abs(optimized masks)
        for key, _, _ in masked_params(self.model):
            self.scores[key] = self.ode.scores[key]

        # reset other parameters
        for p in self.model.parameters():
            p.requires_grad = True

        # freeze masks to be constant
        for _, m, _ in masked_params(self.model):
            m.requires_grad = False

        # get 0-1 / soft masks
        if isinstance(sparsity, list):
            self._local_mask(sparsity)
        else:
            self._global_mask(sparsity)

        # # Adjust BN layers

        tuning_bn = False
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.reset_running_stats()
                tuning_bn = True
                print('Warning: reset BN to (mean, std) = (0, 1) ...')
                pass
        if tuning_bn:
            for i in range(1):
                _, xs = next(enumerate(self.dataloader))
                if hasattr(xs, 'items'):
                    x = {k: v.to(self.device) for k, v in xs.items()}
                else:
                    x, _ = xs
                    x = x.to(self.device)
                _ = self.model(x)
            print('Warning: BN-layers adjusted successfully ...')


class REG(Pruner):
    def __init__(self, model, N=15, r=0.9, E='CE', beta=2 / 3, gamma=1.1, xi=-.1, lambda_=1.0, weight_decay=0.0,
                 pruner_lr=1e-3):
        super().__init__(model)

        from Methods.lzero import hard, penalty

        self.N = N
        r = float(r)
        assert r < 1.0, 'r should less than 1'
        self.r = r
        self.hard = hard
        self.penalty = penalty
        self.E = E
        self.wd = float(weight_decay)
        self.lambda_ = float(lambda_)
        self.lr = float(pruner_lr)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.xi = float(xi)
        self.start = None
        self.init_funcs()

    def init_funcs(self, ):
        if self.E == 'CE':
            self.energy_func = classification_helper()
        elif self.E == 'Trans':
            self.energy_func = transformers_helper()
        else:
            raise NotImplementedError

    @func_timer
    def score(self, criterion, dataloader, device, testloader):
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.testloader = testloader
        for _, m, _ in masked_params(self.model):
            m.grad = None
            m.requires_grad = True
        self.optim = torch.optim.Adam([m for _, m, _ in masked_params(self.model)], lr=self.lr)

    def _global_mask(self, sparsity):

        global_scores = torch.cat([s.flatten() for s in self.scores.values()])
        prune_num = int((1 - sparsity) * global_scores.numel())
        if not prune_num < 1:
            threshold, _ = global_scores.kthvalue(k=prune_num)
            for key, mask, _ in masked_params(self.model):
                score = self.scores[key]
                zero = torch.tensor([0.]).to(mask)
                one = torch.tensor([1.]).to(mask)
                mask.copy_(torch.where(score.le(threshold), zero, one))

    def _local_mask(self, sparsity):

        for (key, mask, _), s in zip(masked_params(self.model), sparsity):
            score = self.scores[key]
            prune_num = int((1 - s) * score.numel())
            if not prune_num < 1:
                threshold, _ = score.flatten().kthvalue(k=prune_num)
                zero = torch.tensor([0.]).to(mask)

                one = torch.tensor([1.]).to(mask)
                mask.copy_(torch.where(score.le(threshold), zero, one))

    def _local_sparsity_l0(self):
        all_p = []
        s_p = []
        for _, m, _ in masked_params(self.model):
            all_p.append(m.numel())
            s_p.append(self.hard(m, deterministic=True).sum())
        return [s / a for s, a in zip(s_p, all_p)]

    def _global_sparsity_l0(self):
        all_p = 0
        s_p = 0
        for _, m, _ in masked_params(self.model):
            all_p += m.numel()
            s_p += self.hard(m, deterministic=True).sum()
        return s_p / all_p

    def sparsity_l0(self, sparsity):
        if isinstance(sparsity, list):
            return self._local_sparsity_l0()
        else:
            return self._global_sparsity_l0()

    @func_timer
    def mask(self, sparsity):
        # freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # allow masks to have gradient

        Normal_ins = Normal(self.r / (1 - self.r), 0.01)
        for _, m, _ in masked_params(self.model):
            m.grad = None
            m.requires_grad = True
            m.data.copy_(Normal_ins.sample(m.shape).log())
        print(f'init m as mean {np.log(self.r / (1 - self.r))}')

        if self.start is None:
            if isinstance(sparsity, list):
                self.start = [self.r] * len(sparsity)
            else:
                self.start = self.r

        self.end = sparsity
        now_sparsity = self.sparsity_l0(self.start)

        for itr in range(self.N):
            break_flag = True
            if isinstance(now_sparsity, list):
                for now_s, target_s in zip(now_sparsity, sparsity):
                    if now_s > target_s:
                        break_flag = False
                        break
            else:
                break_flag = now_sparsity < sparsity
            if break_flag:
                print('break at itr ', itr, now_sparsity)
                break
            _, xs = next(enumerate(self.dataloader))
            if isinstance(xs, dict):
                x = {k: v.to(self.device) for k, v in xs.items()}
                y = None
            else:
                x, y = xs
                x, y = x.to(self.device), y.to(self.device)

            mask_procer = load_mask_procer("l0", beta=self.beta, xi=self.xi, gamma=self.gamma,
                                           deterministic=False)
            reg_mask_proc_hooks(itr, sparsity, self.model, mxp=True, structural=False, mask_dim=1,
                                mask_procer=mask_procer)
            E = self.energy_func(self.model, x, y)
            E_value = E.item()
            for _, m, p in masked_params(self.model):
                E += self.lambda_ * self.penalty(m, p, weight_decay=self.wd, lambda_=self.lambda_)

            self.optim.zero_grad()
            E.backward()
            self.optim.step()
            now_sparsity = self.sparsity_l0(self.start)
            print()
            print(f'No.{itr}: Sparsity is {now_sparsity}  E is {E_value} E+P is {E.item()}')
            print()

        for k, m, _ in masked_params(self.model):
            self.scores[k] = m.clone().data

        for p in self.model.parameters():
            p.requires_grad = True

        # freeze masks to be constant
        for _, m, _ in masked_params(self.model):
            m.requires_grad = False

        # get 0-1 / soft masks
        if isinstance(sparsity, list):
            self._local_mask(sparsity)
        else:
            self._global_mask(sparsity)

        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.reset_running_stats()
                print('Warning: reset BN to (mean, std) = (0, 1) ...')
                pass

        for i in range(1):
            _, (x, _) = next(enumerate(self.dataloader))
            x = x.to(self.device)
            _ = self.model(x)
        print('Warning: BN-layers adjusted successfully ...')
