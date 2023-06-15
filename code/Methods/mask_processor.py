from functools import partial

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm as Gaussian

sns.set_style('darkgrid')

from Methods.utils import is_masked_module, masked_params


# ======================================= Helper functions ============================================ #
def _expand_struct_weight_mask(mask, param, mask_dim=1):
    if param.dim() == 4:
        view_shape = (-1, 1, 1, 1) if mask_dim == 0 else (1, -1, 1, 1)
    else:
        view_shape = (-1, 1) if mask_dim == 0 else (1, -1)

    mask_expanded = mask.view(view_shape).expand_as(param)

    return mask_expanded


def _calc_struct_param(param, mask_dim=1):
    if param.dim() == 4:
        unmasked_dims = (1, 2, 3) if mask_dim == 0 else (0, 2, 3)
        param = param.abs().sum(unmasked_dims)
    elif param.dim() == 2:
        unmasked_dims = (1) if mask_dim == 0 else (0)
        param = param.abs().sum(unmasked_dims)
    else:
        pass

    return param


def _calc_global_mask_vec(model, mxp=True, structural=False, mask_dim=1):
    if mxp:
        if structural:
            ## (mxp, struct)
            global_mask_vec = []
            for _, m, p in masked_params(model):
                p.detach_()
                p_struct = _calc_struct_param(p, mask_dim)
                global_mask_vec += [(m * p_struct).flatten().abs()]

            global_mask_vec = torch.cat(global_mask_vec)

        else:
            ## (mxp, un-struct)
            global_mask_vec = torch.cat([(m * p.detach()).flatten() for _, m, p in masked_params(model)])

    else:
        ## (m, struct) or (m, un-struct)
        global_mask_vec = torch.cat([m.flatten() for _, m, _ in masked_params(model)])

    return global_mask_vec.abs()


def _calc_local_mask_vec(mask, param, mxp=True, structural=False, mask_dim=1):
    param.detach_()

    if mxp:
        if structural:
            ## (mxp, struct)
            param_struct = _calc_struct_param(param, mask_dim)
            local_mask_vec = (mask * param_struct).flatten()

        else:
            ## (mxp, un-struct)
            local_mask_vec = (mask * param).flatten()

    else:
        ## (m, un-struct) or (m, struct)
        local_mask_vec = mask.flatten()

    return local_mask_vec.abs()


# ======================================= Mask processors ============================================ #

class MaskProcessor(nn.Module):
    def __init__(self, one_x_m=False, mask_dim=None, local=False, eps=0.99, ratio=0.99, mxp=False):
        super(MaskProcessor, self).__init__()
        self.one_x_m = one_x_m
        self.mask_dim = mask_dim
        self.structural = mask_dim in [0, 1]
        self.local = local
        self.eps = eps
        self.ratio = ratio
        self.mxp = mxp

    def _calc_ctx_vec(self, mask, param, global_mask_vec):
        if self.local:
            ctx_vec = _calc_local_mask_vec(mask, param, mxp=self.mxp, structural=self.structural,
                                           mask_dim=self.mask_dim)
        else:
            ctx_vec = global_mask_vec

        return ctx_vec

    def _calc_ctx_mask(self, mask, param):
        if self.mxp:
            param.detach_()
            if self.structural:
                param_struct = _calc_struct_param(param, mask_dim=self.mask_dim)
                ctx_mask = mask * param_struct
            else:
                ctx_mask = mask * param
        else:
            ctx_mask = mask

        return ctx_mask.abs()

    def forward(self, itr, mask, param, sparsity, global_mask_inf):
        raise NotImplementedError


class Identity(MaskProcessor):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def forward(self, itr, mask, param, sparsity, global_mask_inf):
        return mask


class OneHot(MaskProcessor):
    def __init__(self, one_x_m=False, mask_dim=None, local=False, eps=0.99, ratio=0.99, mxp=False):
        super(OneHot, self).__init__(one_x_m, mask_dim, local, eps, ratio, mxp)

    def forward(self, itr, mask, param, sparsity, global_mask_inf):

        ctx_vec = self._calc_ctx_vec(mask, param, global_mask_inf['vec'])

        prune_num = int((1 - sparsity) * ctx_vec.numel())
        if prune_num <= 1:
            return mask
        if self.local:
            threshold, _ = ctx_vec.kthvalue(k=prune_num)
        else:
            threshold = global_mask_inf['threshold']
        zero = torch.tensor([0.]).to(mask)
        one = torch.tensor([1.]).to(mask)

        ctx_mask = self._calc_ctx_mask(mask, param)
        mask_proced = mask * torch.where(ctx_mask.le(threshold), zero, one)

        if not self.one_x_m:
            mask_proced /= mask.data.clone().detach()

        return mask_proced


class OneHotHard(MaskProcessor):
    def __init__(self, one_x_m=False, mask_dim=None, local=False, eps=0.99, ratio=0.99, mxp=False):
        super(OneHotHard, self).__init__(one_x_m, mask_dim, local, eps, ratio, mxp)

    def forward(self, itr, mask, param, sparsity, global_mask_inf):

        ctx_vec = self._calc_ctx_vec(mask, param, global_mask_inf['vec'])

        prune_num = int((1 - sparsity) * ctx_vec.numel())
        if prune_num <= 1:
            return mask
        if self.local:
            threshold, _ = ctx_vec.kthvalue(k=prune_num)
        else:
            threshold = global_mask_inf['threshold']
        zero = torch.tensor([0.]).to(mask)
        one = torch.tensor([1.]).to(mask)

        ctx_mask = self._calc_ctx_mask(mask, param)
        mask_proced = torch.where(ctx_mask.le(threshold), zero, one)

        if self.one_x_m:
            mask.data *= mask_proced
        else:
            mask.data = mask_proced

        return mask


class QuantSigmoid(MaskProcessor):
    def __init__(self, one_x_m=False, mask_dim=None, local=False, eps=0.99, ratio=0.99, mxp=False):
        super(QuantSigmoid, self).__init__(one_x_m, mask_dim, local, eps, ratio, mxp)

    def forward(self, itr, mask, param, sparsity, global_mask_inf):
        ctx_vec = self._calc_ctx_vec(mask, param, global_mask_inf['vec'])

        ## [ctx_vec.(ratio * qt), ctx_vec.qt] --> [logit(1 - eps), logit(eps)]
        qt = 1 - sparsity
        eps = torch.Tensor([self.eps])

        ub_orig = ctx_vec.quantile(q=qt)
        lb_orig = ctx_vec.quantile(q=qt * self.ratio)
        ub = eps.logit().to(mask)
        lb = (1 - eps).logit().to(mask)

        ctx_mask = self._calc_ctx_mask(mask, param)

        normalized_mask = (ctx_mask - lb_orig + 1e-12) / (ub_orig - lb_orig + 1e-12)
        mask_proced = (normalized_mask * (ub - lb) + lb).sigmoid()

        if self.one_x_m:
            mask_proced = mask_proced.detach() * mask

        return mask_proced


class GaussSigmoid(MaskProcessor):

    def __init__(self, one_x_m=False, mask_dim=None, local=False, eps=0.99, ratio=0.99, mxp=False):
        super(GaussSigmoid, self).__init__(one_x_m, mask_dim, local, eps, ratio, mxp)

    def forward(self, itr, mask, param, sparsity, global_mask_inf):
        ctx_vec = self._calc_ctx_vec(mask, param, global_mask_inf['vec'])

        ## [mean + Gauss-qt(ratio * qt) * std, mean + Gauss-qt(qt) * std]
        ## --> [logit(1 - eps), logit(eps)]
        qt = 1 - sparsity + 1e-12
        eps = torch.Tensor([self.eps])

        mean, std = ctx_vec.mean(), ctx_vec.std()
        # qt_l, qt_u = torch.Tensor(Gaussian.ppf(q=torch.cat([(qt * self.ratio), qt]).cpu())).to(mask)
        qt_l, qt_u = torch.Tensor(Gaussian.ppf(q=[qt.cpu() * self.ratio, qt.cpu()])).to(mask)
        qt_l, qt_u = max(- 1e12, qt_l), max(- 1e12, qt_u)

        ub_orig = mean + qt_u * std
        lb_orig = mean + qt_l * std
        ub = eps.logit().to(mask)
        lb = (1 - eps).logit().to(mask)

        ctx_mask = self._calc_ctx_mask(mask, param)

        normalized_mask = (ctx_mask - lb_orig + 1e-12) / (ub_orig - lb_orig + 1e-12)
        mask_proced = (normalized_mask * (ub - lb) + lb).sigmoid()

        if self.one_x_m:
            mask_proced = mask_proced.detach() * mask

        return mask_proced


class Lzero(MaskProcessor):
    def __init__(self, beta=2 / 3, gamma=-0.1, xi=1.1, deterministic=False):
        super(Lzero, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.xi = xi
        self.deter = deterministic

    def forward(self, itr, mask, param, sparsity, global_mask_vec):
        from Methods.lzero import hard
        return hard(mask, beta=self.beta, gamma=self.gamma, xi=self.xi, deterministic=self.deter)


# ==================================================================================================== #


def load_mask_procer(mask_proc_option, **kwargs):
    mask_procer_zoo = {
        "Id": Identity,
        "qt": partial(QuantSigmoid, one_x_m=False),
        "qtm": partial(QuantSigmoid, one_x_m=True),
        "gau": partial(GaussSigmoid, one_x_m=False),
        "gaum": partial(GaussSigmoid, one_x_m=True),
        'oh': partial(OneHot, one_x_m=False),
        'ohm': partial(OneHot, one_x_m=True),
        'ohh': partial(OneHotHard, one_x_m=False),
        'ohhm': partial(OneHotHard, one_x_m=True),
        "l0": Lzero
    }
    mask_procer = mask_procer_zoo[mask_proc_option](**kwargs)

    return mask_procer


def reg_mask_proc_hooks(itr, sparsity, model,
                        mxp, structural, mask_dim,
                        mask_procer):
    global_mask_vec = _calc_global_mask_vec(model, mxp, structural, mask_dim)

    prune_num = int((1 - sparsity) * global_mask_vec.numel())
    if prune_num > 1:
        threshold, _ = global_mask_vec.kthvalue(k=prune_num)
    else:
        threshold = None
    global_mask_inf = {
        'vec': global_mask_vec,
        'threshold': threshold
    }

    def _mask_proc(module, input, output):

        if hasattr(module, 'weight_mask'):
            if structural:
                mask_proced = mask_procer(itr, module.weight_mask, module.weight, sparsity, global_mask_inf)
                weight_mask_expanded = _expand_struct_weight_mask(mask_proced, module.weight, mask_dim)

                w = module.weight * weight_mask_expanded

            else:
                mask_proced = mask_procer(itr, module.weight_mask, module.weight, sparsity, global_mask_inf)
                w = module.weight * mask_proced
        else:
            w = module.weight

        if hasattr(module, 'bias_mask'):
            mask_proced = mask_procer(itr, module.bias_mask, module.bias, sparsity, global_mask_inf)
            b = module.bias * mask_proced
        else:
            b = module.bias

        input = input[0]

        if isinstance(module, (nn.Linear, nn.Linear)):
            output = F.linear(input, w, b)
        elif isinstance(module, (nn.Conv2d, nn.Conv2d)):
            if module.padding_mode != 'zeros':
                from torch.nn.modules.utils import _pair
                output = F.conv2d(F.pad(input, module._padding_repeated_twice, mode=module.padding_mode),
                                  w, b,
                                  module.stride, _pair(0), module.dilation, module.groups)
            else:
                output = F.conv2d(input, w, b,
                                  module.stride, module.padding, module.dilation, module.groups)

        return output

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Linear, nn.Conv2d)) and is_masked_module(m):
            if hasattr(m, 'fwd_hook'):
                m.fwd_hook.remove()
            m.fwd_hook = m.register_forward_hook(_mask_proc)


if __name__ == '__main__':
    pass

    ## 1. Visualization for each mask-processors
    N = 1000
    sparsity = torch.Tensor([0.2])
    x = torch.randn(N)
    x /= x.norm()
    x *= sparsity

    df = pd.DataFrame(data=dict(x=x.numpy().astype(np.float16)))

    kwargs = dict(local=True, eps=0.9, ratio=0.8)

    mask_proc = load_mask_procer('oh', **kwargs)
    x_proced = mask_proc(None, x, None, sparsity, None)
    x_proced /= x
    df_oh = pd.DataFrame(data=dict(x=x_proced.numpy().astype(np.float16)))

    mask_proc = load_mask_procer('gau', **kwargs)
    x_proced = mask_proc(None, x, None, sparsity, None)
    df_gau = pd.DataFrame(data=dict(x=x_proced.numpy().astype(np.float16)))

    mask_proc = load_mask_procer('qt', **kwargs)
    x_proced = mask_proc(None, x, None, sparsity, None)
    df_qt = pd.DataFrame(data=dict(x=x_proced.numpy().astype(np.float16)))

    print('Finished computed.')

    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(3 * 2 / 0.618, 3 * 2))
    p = sns.histplot(data=df, x='x', ax=ax[0, 0], stat='probability')
    p.set_title('Origin')
    p = sns.histplot(data=df_oh, x='x', ax=ax[0, 1], stat='probability')
    p.set_title('One hot ')
    p = sns.histplot(data=df_qt, x='x', ax=ax[1, 0], stat='probability')
    p.set_title('Direct quantile matching')
    p = sns.histplot(data=df_gau, x='x', ax=ax[1, 1], stat='probability')
    p.set_title('Gaussian quantile matching')
    plt.savefig('./mask_processor.pdf', dpi=400)
    plt.cla()

    ## 2. Demo: how mask processor hook works
    #
    # class MLP(nn.Module):
    #     def __init__(self, dim):
    #         super(MLP, self).__init__()
    #         self.fc1 = nn.Linear(dim, dim)
    #         self.act = nn.ReLU()
    #         self.fc2 = nn.Linear(dim, dim)
    #
    #     def forward(self, x):
    #         return self.fc2(self.act(self.fc1(x)))
    #
    #
    # device = 'cuda:0'
    #
    # dim = 4
    # model = MLP(dim).to(device)
    # x = torch.randn(7, dim).to(device)
    #
    # from Methods.utils import init_masks_hooks, masked_params, is_masked_module, reg_fwd_hooks
    # import copy
    #
    # init_masks_hooks(model, False)
    # for _, m, _ in masked_params(model):
    #     m.requires_grad = True
    #
    # model2 = copy.deepcopy(model)
    #
    # out = model(x).softmax(0).norm()
    # out2 = model2(x).softmax(0).norm()
    # print(f'out1 - out2 =  {(out - out2).norm()}')
    #
    # structural = False
    # mask_dim = 1
    # local = False
    #
    # normalizer = sum([p.numel() for _, _, p in masked_params(model2)])
    # sparsity = sum([m.norm(p=1) for _, m, _ in masked_params(model2)]) / normalizer
    #
    # kwargs = dict(local=True, eps=0.99, ratio=0.99)
    # mask_procer = load_mask_procer('oh', **kwargs)
    # reg_mask_proc_hooks(None, sparsity, model, structural, mask_dim, mask_procer)
    #
    # out = model(x).softmax(0).norm()
    # out2 = model2(x).softmax(0).norm()
    # print(f'out1 - out2 =  {(out - out2).norm()}')
    #
    # from torch.autograd import grad
    #
    # grads = grad(out2, [m for _, m, _ in masked_params(model2)])
    # print(grads)
