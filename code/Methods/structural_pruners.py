import copy

import torch

from Methods.utils import masked_params
from Utils.misc import func_timer


class StructPruner:
    def __init__(self, model, mask_dim, quant=False):
        self.model = model
        self.mask_dim = mask_dim
        self.scores = {}
        for key, m, _ in masked_params(self.model):
            self.scores[key] = torch.ones_like(m).to(m)

        self.quant = quant
        self.quant_path = None

    def score(self, criterion, dataloader, device, testloader):
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

    def stats(self, model):
        node_sparsities = [m.data.count_nonzero().item() / m.numel()
                           for _, m, p in masked_params(model)
                           if p.dim() in [2, 4]  # only for weight-layers (FC & Conv2d)
                           ]
        node_sparsities = [1.] + node_sparsities if self.mask_dim == 0 else node_sparsities + [1.]

        sparsities = []
        remaining, total = 0, 0
        weight_layer_num = 0
        for _, m, p in masked_params(model):

            if p.dim() in [2, 4]:
                node_size = p.data[0, :].numel() if self.mask_dim == 0 else p.data[:, 0].numel()
                ## param. sparsity = current node sparsity x prev / next node sparsity
                node_size *= node_sparsities[weight_layer_num]
                weight_layer_num += 1

                remaining += m.data.count_nonzero().item() * node_size
                sparsities += [m.data.count_nonzero().item() * node_size / p.numel()]
            else:
                remaining += m.data.count_nonzero().item()
                sparsities += [m.data.count_nonzero().item() / p.numel()]

            total += p.data.numel()

        print('Structural sparsity:', sparsities)

        return remaining, total, remaining / total


class Rand(StructPruner):
    def __init__(self, model, mask_dim):
        super(Rand, self).__init__(model, mask_dim)

    @func_timer
    def score(self, criterion, dataloader, device, testloader):
        for key, m, _ in masked_params(self.model):
            self.scores[key] = torch.randn_like(m).to(m)


class Mag(StructPruner):
    def __init__(self, model, mask_dim):
        super(Mag, self).__init__(model, mask_dim)

    @func_timer
    def score(self, criterion, dataloader, device, testloader):
        for key, m, w in masked_params(self.model):
            score_full = w.detach().abs()

            if w.dim() == 4:
                unmasked_dims = (1, 2, 3) if self.mask_dim == 0 else (0, 2, 3)
                self.scores[key] = score_full.sum(unmasked_dims)
                # self.scores[key] = score_full.mean(unmasked_dims)
            elif w.dim() == 2:
                unmasked_dims = (1) if self.mask_dim == 0 else (0)
                self.scores[key] = score_full.sum(unmasked_dims)
                # self.scores[key] = score_full.mean(unmasked_dims)
            else:
                self.scores[key] = score_full


class MagRand(StructPruner):
    def __init__(self, model, mask_dim):
        super(MagRand, self).__init__(model, mask_dim)

    @func_timer
    def score(self, criterion, dataloader, device, testloader):
        for key, _, p in masked_params(self.model):
            score_full = (p * torch.randn_like(p).to(p)).detach().abs()

            if p.dim() == 4:
                unmasked_dims = (1, 2, 3) if self.mask_dim == 0 else (0, 2, 3)
                self.scores[key] = score_full.sum(unmasked_dims)
            elif p.dim() == 2:
                unmasked_dims = (1) if self.mask_dim == 0 else (0)
                self.scores[key] = score_full.sum(unmasked_dims)
            else:
                self.scores[key] = score_full


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(StructPruner):
    def __init__(self, model, mask_dim):
        super(SNIP, self).__init__(model, mask_dim)

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

        for key in self.scores.keys():
            self.scores[key] = self.scores[key].div(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(StructPruner):
    def __init__(self, model, mask_dim):
        super(GraSP, self).__init__(model, mask_dim)
        self.temp = 200
        self.eps = 1e-10

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
            score_full = self.scores[key].div(norm)

            if p.dim() == 4:
                unmasked_dims = (1, 2, 3) if self.mask_dim == 0 else (0, 2, 3)
                self.scores[key] = score_full.sum(unmasked_dims)
            elif p.dim() == 2:
                unmasked_dims = (1) if self.mask_dim == 0 else (0)
                self.scores[key] = score_full.sum(unmasked_dims)
            else:
                self.scores[key] = score_full


class SynFlow(StructPruner):
    def __init__(self, model, mask_dim):
        super(SynFlow, self).__init__(model, mask_dim)

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

            score_full = torch.clone(p.grad * p).detach().abs()

            if p.dim() == 4:
                unmasked_dims = (1, 2, 3) if self.mask_dim == 0 else (0, 2, 3)
                self.scores[key] = score_full.sum(unmasked_dims)
            elif p.dim() == 2:
                unmasked_dims = (1) if self.mask_dim == 0 else (0)
                self.scores[key] = score_full.sum(unmasked_dims)
            else:
                self.scores[key] = score_full

            p.grad.data.zero_()

        nonlinearize(self.model, signs)


class ODE(StructPruner):
    def __init__(self, model, mask_dim, N=15, r=1, ode_scope='global', E='CE', G='l1',
                 score_option='m', mask_option='one', mask_proc_kwargs={},
                 schedule='lin', rt_schedule='fix', momentum=0, start=None, quant=False, save_dir=None):
        super(ODE, self).__init__(model, mask_dim, quant)
        from Methods.sparsity_ode import SparsityIndexedODE
        self.ode = SparsityIndexedODE(mask_dim, N, r, ode_scope, E, G, score_option, mask_proc_kwargs,
                                      schedule, rt_schedule, momentum)
        self.score_option = score_option
        self.mask_option = mask_option
        self.start = start

    @func_timer
    def score(self, criterion, dataloader, device, testloader):
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device

    def _global_mask(self, sparsity):
        global_scores = torch.cat([s.flatten() for s in self.scores.values()])
        prune_num = int((1 - sparsity) * global_scores.numel())
        if not prune_num < 1:
            threshold, _ = global_scores.kthvalue(k=prune_num)
            for key, mask, _ in masked_params(self.model):
                score = self.scores[key]
                zero = torch.tensor([0.]).to(mask)

                if self.mask_option == 'one':
                    one = torch.tensor([1.]).to(mask)
                    mask.copy_(torch.where(score <= threshold, zero, one))

                elif self.mask_option == 'mask':
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score <= threshold, zero, mask_copy))

                elif self.mask_option == 'sign':
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score <= threshold, zero, mask_copy.sign()))

    def _local_mask(self, sparsity):
        for (key, mask, _), s in zip(masked_params(self.model), sparsity):
            score = self.scores[key]
            prune_num = int((1 - s) * score.numel())
            if not prune_num < 1:
                threshold, _ = score.flatten().kthvalue(k=prune_num)
                zero = torch.tensor([0.]).to(mask)

                if self.mask_option == 'one':
                    one = torch.tensor([1.]).to(mask)
                    mask.copy_(torch.where(score <= threshold, zero, one))

                elif self.mask_option == 'mask':
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score <= threshold, zero, mask_copy))

                elif self.mask_option == 'sign':
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score <= threshold, zero, mask_copy.sign()))

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

        # Adjust BN layers
        self.model.train()
        _, (x, _) = next(enumerate(self.dataloader))
        x = x.to(self.device)
        _ = self.model(x)
        print('Warning: BN-layers adjusted successfully.')
