import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')
import matplotlib
import copy

matplotlib.use('Agg')

from Utils.misc import func_timer
from Methods.utils import masked_params, reg_fwd_hooks, classification_helper
from Methods.mask_processor import load_mask_procer, reg_mask_proc_hooks, _calc_global_mask_vec


# quantile_record_helper(d_x,'mask')
# quantile_record_helper(d_theta,'theta')
# quantile_record_helper(d_x_theta,'m_t')

def quantile_print_helper(d: dict):
    print(
        f"m < 0 rate = {d['mle0']}%, "
        f"m > 1 rate = {d['mge1']}%.\n"
        f"qt 1% =  {d['qt1']}, "
        f"qt 10% = {d['qt10']},\n"
        f"qt 50% = {d['qt50']},\n"
        f"qt 90% = {d['qt90']}, "
        f"qt 99% = {d['qt99']}.\n")


def quantile_record_helper(d: dict, flag: str):
    print(f"{flag} m 0 {d['mle0']:.04f}")
    print(f"{flag} m 1 {d['mge1']:.04f}")
    print(f"{flag} e 0 {d['meq0']:.04f}")
    print(f"{flag} qt 1 {d['qt1']:.04f}")
    print(f"{flag} qt 10 {d['qt10']:.04f}")
    print(f"{flag} qt 50 {d['qt50']:.04f}")
    print(f"{flag} qt 90 {d['qt90']:.04f}")
    print(f"{flag} qt 99 {d['qt99']:.04f}")


def print_stat(itr, N, sparsity, model, mask_procer, model_size,
               track_m_order, track_p_order, track_mp_order, structural=False):
    print(f'Forward itr [{itr + 1} / {N}] ...')

    masks_flatten = torch.cat([(mask_procer(itr, m, p, sparsity, model)).flatten().detach()
                               for _, m, p in masked_params(model)]).flatten()
    violation_0 = masks_flatten.le(0).sum().item() / model_size * 100
    violation_1 = masks_flatten.ge(1).sum().item() / model_size * 100
    print(f'm < 0 rate = {violation_0}%, '
          f'm > 1 rate = {violation_1}%.\n'
          f'qt 1% = {torch.quantile(masks_flatten, 0.01)}, '
          f'qt 10% = {torch.quantile(masks_flatten, 0.1)},\n'
          f'qt 50% = {torch.quantile(masks_flatten, 0.5)},\n'
          f'qt 90% = {torch.quantile(masks_flatten, 0.9)}, '
          f'qt 99% = {torch.quantile(masks_flatten, 0.99)}.\n'
          )

    if not structural:
        mask_x_para_flatten = torch.cat(
            [(m * p).flatten().detach() for _, m, p in masked_params(model)]).flatten()
        mask_x_para_order = mask_x_para_flatten.argsort()
        track_mp_order.append(mask_x_para_order[:100].cpu().numpy())
        np.savetxt('track_mp_sys.csv', np.array(track_mp_order), delimiter=',')

    mask_order = masks_flatten.argsort()
    track_m_order.append(mask_order[:100].cpu().numpy())
    np.savetxt('track_m_sys.csv', np.array(track_m_order), delimiter=',')

    para_flatten = torch.cat([p.flatten().detach() for _, _, p in masked_params(model)]).flatten()
    para_order = para_flatten.argsort()
    track_p_order.append(para_order[:100].cpu().numpy())
    np.savetxt('track_p_sys.csv', np.array(track_p_order), delimiter=',')

    return track_m_order, track_p_order, track_mp_order


class StatTracker():
    def __init__(self, stat_names, save_dir=None):
        self.save_dir = save_dir
        self.stat_names = stat_names
        self.stats = {stat_name: [] for stat_name in stat_names}
        print('\n\tODE statistic tracker initialized successfully ...')
        print(f'\tstats-to-track = {stat_names}.\n')

    def update_stats(self, stat_vals):
        for key in self.stats.keys():
            self.stats[key].append(stat_vals[key])

    def save_stats(self):
        df = pd.DataFrame(data=self.stats)
        if self.save_dir:
            df.to_csv(f'{self.save_dir}/stats.csv')
        else:
            print('Warning: save_dir does not exists. Skip saving ...')

    def show_stats(self, stat_vals, ncols=3):
        print('\n')
        msg = []
        for i, key in enumerate(self.stats.keys()):
            msg.append('{0:<18} = {1:.5f}'.format(key, stat_vals[key]))
            if (i + 1) % ncols == 0 or (i + 1) == len(self.stats.keys()):
                print('\t' + ', '.join(msg))
                msg = []


class SparsityIndexedODE:

    def __init__(self, mask_dim=None, N=15, r=1.1, ode_scope='global',
                 E='CE', G='l2', score_option='mp', mask_proc_kwargs={},
                 schedule='lin', rt_schedule='fix', momentum=0.,
                 save_dir=None, save_ckpt=False, ):
        self.N = int(N)  # num. of bins
        assert r > 1, f'Error: r_min = {r} <= 1!'
        self.r_min = r  # minimal localization radius
        self.r_max = 4
        self.r = r

        self.E = E  # Energy to preserve
        self.G = G  # sparsity to reduce
        self.score_option = score_option
        self.mask_proc_option = mask_proc_kwargs['mask_proc_option']
        self.mask_proc_eps = mask_proc_kwargs['mask_proc_eps']
        self.mask_proc_ratio = mask_proc_kwargs['mask_proc_ratio']
        self.mask_proc_score_option = mask_proc_kwargs['mask_proc_score_option']
        self.mask_proc_mxp = mask_proc_kwargs['mask_proc_mxp']

        self.schedule = schedule
        self.rt_schedule = rt_schedule
        if rt_schedule in ['hess', 'ada']:
            assert schedule == rt_schedule, '\n\nWarning: dt & r_t should incompatible!\n\n'

        self.momentum = momentum
        self.local_ode = ode_scope == 'local'
        self.init_funcs()

        self.structural = mask_dim in [0, 1]
        self.mask_dim = mask_dim
        if self.structural:
            print('\tUsing structural ODE ...')
        else:
            print('\tUsing unstructural ODE ...')

        self.save_dir = save_dir
        self.save_ckpt = save_ckpt
        if save_dir:
            print('\tWarning: tracking ODE stats ...')
            if save_ckpt:
                self.G_mile_stones = list(reversed([0.5, 0.28, 0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.02]))
                print(f'\n\nWarning: saving intermediate model ckpts sp in {self.G_mile_stones} ...\n\n')

    # ======================================= General helpers =================================================== #

    def _save_ckpt(self, itr, model, G, G_ideal):
        model_tmp = copy.deepcopy(model)
        self._permanent_polarization(model_tmp, G)

        ckpt = dict(
            model=model_tmp.state_dict(),
            scores=self.scores,
            sparsity=G,
            itr=itr + 1,
        )
        torch.save(ckpt, f'{self.save_dir}/sp{G_ideal}.pkl')
        print(f'\tODE checkpoint of sparsity = {G_ideal} with reset fwd-hooks saved successfully ...\n')

    def _update_stat_tracker(self, itr, dt, r_t, G, E, delta, neg_dE, neg_dG, verbose=True):

        dt = dt[0] if isinstance(dt, list) else dt

        stat_vals = {
            'itr': int(itr + 1),
            'dt': dt,
            'r_t': r_t,
            'sparsity': G.item(),
            'energy': E.item(),
            'cos(delta, - dE)': self._calc_cosine(delta, neg_dE).item(),
            'cos(delta, - dG)': self._calc_cosine(delta, neg_dG).item(),
            'cos(dG, dE)': self._calc_cosine(neg_dE, neg_dG).item(),
            'norm(dG)': self._calc_norm(neg_dG).item(),
            'norm(dE)': self._calc_norm(neg_dE).item(),
        }
        self.stat_tracker.update_stats(stat_vals)
        if verbose:
            self.stat_tracker.show_stats(stat_vals, ncols=3)

    def init_funcs(self, ):
        if self.E == 'CE':
            self.energy_func = classification_helper()
        else:
            raise NotImplementedError

    def _calc_norm(self, x):
        x_norm = torch.sqrt(sum([xi.norm() ** 2 for xi in x]) + 1e-12)
        return x_norm

    def _calc_inner_prod(self, x, y):
        x_dot_y = sum([(xi * yi).sum() for xi, yi in zip(x, y)])
        return x_dot_y

    def _calc_cosine(self, x, y):
        x_norm = self._calc_norm(x)
        y_norm = self._calc_norm(y)
        x_dot_y = self._calc_inner_prod(x, y)

        return x_dot_y / (x_norm * y_norm + 1e-12)

    def _permanent_polarization(self, model, sparsity):
        ## load polarizor
        mask_procer_kwargs = dict(mask_dim=self.mask_dim, local=self.local_prune, eps=self.mask_proc_eps,
                                  ratio=self.mask_proc_ratio, mxp=self.mask_proc_mxp)
        mask_procer = load_mask_procer(self.mask_proc_option, **mask_procer_kwargs)

        ## polarize mask values
        global_mask_vec = _calc_global_mask_vec(model, self.mask_proc_mxp, self.structural, self.mask_dim)
        prune_num = int((1 - sparsity) * global_mask_vec.numel())
        if prune_num > 1:
            threshold, _ = global_mask_vec.kthvalue(k=prune_num)
        else:
            threshold = None
        global_mask_inf = {
            'vec': global_mask_vec,
            'threshold': threshold
        }
        for _, m, p in masked_params(model):
            mask_proced = mask_procer(None, m, p, sparsity, global_mask_inf)
            m.data = mask_proced

        ## polarizor hook --> mask mulitplicative hook
        reg_fwd_hooks(model, self.structural, self.mask_dim)

        print('\tMasks permanently polarized successfully ...')

    # =========================================== Calc. ODE-Stats =============================================== #

    def _calc_r_t(self, itr, neg_dE):
        # here r_t is not as same as paper written
        # r_t in paper is r_t_code * g_norm
        if self.rt_schedule == 'fix':
            r_t = self.r_min

        elif self.rt_schedule == 'invexp':
            if self.local_ode:
                raise NotImplementedError
            else:
                # itr starts from 0
                dE_norm = self._calc_norm(neg_dE)
                ratio = self.r_min / self.r_max
                r_t = self.r_max * ratio ** (1 - (itr + 1) / self.N)

        elif self.rt_schedule == 'hess':
            if self.local_ode:
                raise NotImplementedError
            else:
                dE_norm = self._calc_norm(neg_dE)
                r_t = 1 + (self.r_max - 1) * dE_norm
                r_t = max(self.r_min, r_t.item())

        else:
            raise NotImplementedError

        return r_t

    def _calc_dt(self, itr, start, end, G=None, neg_dE=None, G_local=None):
        self.local_prune = isinstance(start, list)

        if self.schedule == 'lin':
            if self.local_prune:
                dt = [(s - e) / self.N for s, e in zip(start, end)]
            else:
                dt = (start - end) / self.N

        elif self.schedule == 'exp':
            if self.local_prune:
                dt = [e * ((s / e) ** (1 - itr / self.N) - (s / e) ** (1 - (itr + 1) / self.N))
                      for s, e in zip(start, end)]
            else:
                # itr starts from 0
                ratio = start / end
                dt = end * (ratio ** (1 - itr / self.N) - ratio ** (1 - (itr + 1) / self.N))

        elif self.schedule == 'invexp':
            if self.local_prune:
                dt = [s * ((e / s) ** (1 - (itr + 1) / self.N) - (e / s) ** (1 - itr / self.N))
                      for s, e in zip(start, end)]
            else:
                # itr starts from 0
                ratio = end / start
                dt = start * (ratio ** (1 - (itr + 1) / self.N) - ratio ** (1 - itr / self.N))

        # ========================== use with un-fixed r_t schedule ======================= #

        elif self.schedule == 'hess':
            # dt = const / norm(dE)
            dE_norm = torch.sqrt(sum([de.norm() ** 2 for de in neg_dE]) + 1e-12).item()
            if self.local_prune:
                dt = [(G.item() - e) / ((self.N - itr) * dE_norm + 1e-12)
                      for e in end]
            else:
                dt = (G.item() - end) / ((self.N - itr) * dE_norm + 1e-12)

        # ================================================================================= #

        return dt

    def _calc_G(self, model, eps=1e-12):
        G_local = []

        ## Local ODE ##
        if self.local_ode:
            if self.G == 'l1':
                # sparsity = mask l1-norm
                if self.structural:
                    G = []
                    for _, m, p in masked_params(model):
                        if p.dim() in [2, 4]:
                            node_size = p.data[0, :].numel() if self.mask_dim == 0 else p.data[:, 0].numel()
                            G += [m.norm(p=1) * node_size / p.numel()]
                        else:
                            G += [m.norm(p=1) / p.numel()]

                else:
                    G = [m.norm(p=1) / p.numel() for _, m, p in masked_params(model)]

            elif self.G == 'l2':
                # sparsity = mask l2-norm
                if self.structural:
                    G = []
                    for _, m, p in masked_params(model):
                        if p.dim() in [2, 4]:
                            node_size = p.data[0, :].numel() if self.mask_dim == 0 else p.data[:, 0].numel()
                            G += [m.norm(p=2) * (node_size ** 0.5) / p.numel() ** 0.5]
                        else:
                            G += [m.norm(p=2) / p.numel() ** 0.5]

                else:
                    G = [m.norm(p=2) / p.numel() ** 0.5 for _, m, p in masked_params(model)]

            else:
                raise NotImplementedError

            return sum(G), G

        ## Global ODE ##
        else:
            if self.G == 'l1':
                # sparsity = mask l1-norm
                normalizer = sum([p.numel() for _, _, p in masked_params(model)])
                if self.structural:
                    G = 0.
                    for _, m, p in masked_params(model):
                        if p.dim() in [2, 4]:
                            node_size = p.data[0, :].numel() if self.mask_dim == 0 else p.data[:, 0].numel()
                            G_tmp = m.norm(p=1) * node_size
                        else:
                            G_tmp = m.norm(p=1)

                        G += G_tmp
                        G_local.append(G_tmp / p.numel())

                    G /= normalizer
                else:
                    G = sum([m.norm(p=1) for _, m, _ in masked_params(model)]) / normalizer

            elif self.G == 'l2':
                # sparsity = mask l2-norm
                normalizer = sum([p.numel() for _, _, p in masked_params(model)]) ** 0.5
                if self.structural:
                    G = 0.
                    for _, m, p in masked_params(model):
                        if p.dim() in [2, 4]:
                            node_size = p.data[0, :].numel() if self.mask_dim == 0 else p.data[:, 0].numel()
                            G_tmp = (m.norm(p=2) ** 2) * node_size
                        else:
                            G_tmp = m.norm(p=2) ** 2

                        G += G_tmp
                        G_local.append(G_tmp ** 0.5 / p.numel() ** 0.5)

                    G = torch.sqrt(G + eps) / normalizer
                else:
                    G = torch.sqrt(sum([m.norm(p=2) ** 2 for _, m, _ in masked_params(model)]) + eps) / (
                            normalizer + eps)

            else:
                raise NotImplementedError

            return G, G_local

    def _calc_neg_dG(self, model):
        G, G_local = self._calc_G(model)
        neg_dG = torch.autograd.grad(- G, [m for _, m, _ in masked_params(model)], allow_unused=True)
        G.detach_()
        return G, neg_dG, G_local

    def _calc_neg_dE(self, itr, G, model, x, y):
        ## Mark down un-polarized mask values
        if self.mask_proc_option in ['ohh', 'ohhm']:
            masks_prev = [m.data.clone() for _, m, _ in masked_params(model)]

        ## Polarize masks
        mask_procer_kwargs = dict(mask_dim=self.mask_dim, local=self.local_prune, eps=self.mask_proc_eps,
                                  ratio=self.mask_proc_ratio, mxp=self.mask_proc_mxp)
        self.mask_procer = load_mask_procer(self.mask_proc_option, **mask_procer_kwargs)
        reg_mask_proc_hooks(itr, G, model, self.mask_proc_mxp, self.structural, self.mask_dim, self.mask_procer)

        ## Calc. energy with porlarized model
        E = self.energy_func(model, x, y)
        neg_dE = torch.autograd.grad(- E, [m for _, m, _ in masked_params(model)], allow_unused=True)
        E.detach_()

        ## Un-polarize masks for ODE update
        if self.mask_proc_option in ['ohh', 'ohhm']:
            for i, (_, m, _) in enumerate(masked_params(model)):
                m.data.copy_(masks_prev[i])

        return E, neg_dE

    def _calc_delta(self, r_t, vec_G, vec_E, eps=1e-12):
        if self.local_ode:
            delta = [None] * len(vec_G)
            for layer, (g, e) in enumerate(zip(vec_G, vec_E)):
                g_norm = g.norm()
                e_norm = e.norm()
                gxe = (g * e).sum()

                x = torch.sqrt((r_t ** 2 - 1) / ((g_norm * e_norm) ** 2 - gxe ** 2 + eps) + eps)
                y = (1 - x * gxe) / (g_norm ** 2 + eps)
                delta[layer] = (x * e + y * g).clone()

        else:
            G_norm = torch.sqrt(sum([g.norm() ** 2 for g in vec_G]) + eps)
            E_norm = torch.sqrt(sum([e.norm() ** 2 for e in vec_E]) + eps)
            GxE = sum([(g * e).sum() for g, e in zip(vec_G, vec_E)])

            x = torch.sqrt((r_t ** 2 - 1 + eps) / ((G_norm * E_norm) ** 2 - GxE ** 2 + eps) + eps)
            y = (1 - x * GxE) / (G_norm ** 2 + eps)
            delta = [(x * e + y * g).clone() for g, e in zip(vec_G, vec_E)]

        return delta

    # ============================================= Core functions ============================================= #

    @torch.no_grad()
    def _update_scores(self, itr, model):

        assert self.mask_proc_score_option == 'Id', 'importance scores are based on un-polarized (i.e. Id-polarization) masks'
        mask_procer_kwargs = dict(mask_dim=self.mask_dim, local=self.local_prune, eps=self.mask_proc_eps,
                                  ratio=self.mask_proc_ratio, mxp=self.mask_proc_mxp)
        mask_procer_for_score = load_mask_procer(self.mask_proc_score_option, **mask_procer_kwargs)
        print(f'Using {self.mask_proc_score_option} score')

        def _unstructural_score(itr, mask, param, sparsity, score_option, mask_procer):
            if score_option == 'm':
                score_tmp = (mask_procer(itr, mask, param, sparsity, model)).abs().clone()
            elif score_option == 'mp':
                score_tmp = (mask_procer(itr, mask, param, sparsity, model) * param).abs().clone()
            else:
                raise NotImplementedError

            return score_tmp

        def _structural_score(itr, mask, param, sparsity, score_option, mask_procer, mask_dim):
            ## Conv2d
            if param.dim() == 4:
                view_shape = (-1, 1, 1, 1) if mask_dim == 0 else (1, -1, 1, 1)
                unmasked_dims = (1, 2, 3) if mask_dim == 0 else (0, 2, 3)
                mask_proced = mask_procer(itr, mask, param, sparsity, model).view(view_shape).expand_as(p)

            ## FC
            elif param.dim() == 2:
                view_shape = (-1, 1) if mask_dim == 0 else (1, -1)
                unmasked_dims = (1) if mask_dim == 0 else (0)
                mask_proced = mask_procer(itr, mask, param, sparsity, model).view(view_shape).expand_as(p)

            ## FC-bias
            else:
                unmasked_dims = None
                mask_proced = mask_procer(itr, mask, param, sparsity, model)

            if score_option == 'm':
                score_tmp = (mask_proced).abs().clone()
            elif score_option == 'mp':
                score_tmp = (mask_proced * param).abs().clone()
            else:
                raise NotImplementedError

            if unmasked_dims is not None:
                score_tmp = score_tmp.sum(unmasked_dims)

            return score_tmp

        if not hasattr(self, 'scores'):
            self.scores = {key: torch.zeros_like(m).to(m) for key, m, _ in masked_params(model)}

        sparsity = self._calc_G(model)
        for key, m, p in masked_params(model):
            if self.structural:
                score_tmp = _structural_score(itr, m, p, sparsity, self.score_option, mask_procer_for_score,
                                              self.mask_dim)
            else:
                score_tmp = _unstructural_score(itr, m, p, sparsity, self.score_option, mask_procer_for_score)

            self.scores[key] = score_tmp + self.momentum * self.scores[key]

    def _update_masks(self, model, delta, dt):
        for layer, (_, m, _) in enumerate(masked_params(model)):
            if self.local_prune:
                m.data += delta[layer] * dt[layer]
            else:
                m.data += delta[layer] * dt

    @func_timer
    def _one_step_ode(self, itr, start, end, model, G, G_local, neg_dG, neg_dE):

        ## Direct update
        r_t = self._calc_r_t(itr, neg_dE)
        delta = self._calc_delta(r_t, neg_dG, neg_dE)
        dt = self._calc_dt(itr, start, end, G=G, neg_dE=neg_dE, G_local=G_local)

        self._update_masks(model, delta, dt)

        return r_t, delta, dt

    @func_timer
    def discretization(self, start, end, model, dataloader, device, quant_path=None, testloader=None):
        self.model_size = sum([p.numel() for p in model.parameters()])
        self.local_prune = isinstance(start, list)

        for _, m, _ in masked_params(model):
            m.requires_grad = True

        stat_names = ['itr', 'dt', 'r_t', 'sparsity', 'energy',
                      'cos(delta, - dG)', 'cos(delta, - dE)', 'cos(dG, dE)',
                      'norm(dG)', 'norm(dE)']
        self.stat_tracker = StatTracker(stat_names, save_dir=self.save_dir)

        for itr in range(self.N):

            _, xs = next(enumerate(dataloader))
            if hasattr(xs, 'items'):
                x = {k: v.to(device) for k, v in xs.items()}
                y = None
            else:
                x, y = xs
                x, y = x.to(device), y.to(device)

            G, neg_dG, G_local = self._calc_neg_dG(model)
            E, neg_dE = self._calc_neg_dE(itr, G, model, x, y)

            r_t, delta, dt = self._one_step_ode(itr, start, end, model, G, G_local, neg_dG, neg_dE)

            self._update_scores(itr, model)

            self._update_stat_tracker(itr, dt, r_t, G, E, delta, neg_dE, neg_dG)
            self.stat_tracker.save_stats()

            if self.save_ckpt:
                G, _ = self._calc_G(model)
                # first time reach a sparsity milestone
                if len(self.G_mile_stones) >= 1 or G <= self.G_mile_stones[-1]:
                    G_ideal = self.G_mile_stones.pop()
                    self._save_ckpt(itr, model, G, G_ideal)

        ## Permanent mask polarization
        self._permanent_polarization(model, G)
