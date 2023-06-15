import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    Categorize modules
'''


def is_masked_module(m):
    return hasattr(m, 'weight_mask') or hasattr(m, 'bias_mask')


def init_masks_hooks(model, structural=False, mask_dim=0):
    init_masks(model, structural, mask_dim)
    reg_fwd_hooks(model, structural, mask_dim)


def init_masks(model, structural=False, mask_dim=1):
    if structural:
        _init_structural_masks(model, mask_dim)
    else:
        _init_unstructural_masks(model)


def _init_structural_masks(model, mask_dim):
    assert mask_dim in [0, 1]  # 0 = out_dim, 1 = in_dim

    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Linear, nn.Conv2d)):
            weight_mask_structural = torch.ones(m.weight.data.size(mask_dim)).to(m.weight.device)
            m.register_buffer('weight_mask', weight_mask_structural)
            if m.bias is not None:
                m.register_buffer('bias_mask', torch.ones_like(m.bias.data).to(m.bias.device))

    print('\t Structral masks initialized successfully ...')


def _init_unstructural_masks(model):
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Linear, nn.Conv2d)):
            m.register_buffer('weight_mask', torch.ones_like(m.weight.data).to(m.weight.device))
            if m.bias is not None:
                m.register_buffer('bias_mask', torch.ones_like(m.bias.data).to(m.bias.device))

    print('\t Unstructral masks initialized successfully ...')


def reg_fwd_hooks(model, structural, mask_dim):
    def _apply_mask(module, input, output):

        if hasattr(module, 'weight_mask'):
            if structural:
                if module.weight.dim() == 4:
                    view_shape = (-1, 1, 1, 1) if mask_dim == 0 else (1, -1, 1, 1)
                else:
                    view_shape = (-1, 1) if mask_dim == 0 else (1, -1)

                weight_mask_expanded = module.weight_mask.view(view_shape).expand_as(module.weight)
                w = module.weight * weight_mask_expanded

            else:
                w = module.weight * module.weight_mask
        else:
            w = module.weight

        if hasattr(module, 'bias_mask'):
            b = module.bias * module.bias_mask
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
            m.fwd_hook = m.register_forward_hook(_apply_mask)

    hook_type = 'Structural' if structural else 'Unstructural'
    print(f'\t {hook_type} forward-hooks registered successfully ...')


def free_all_hooks_and_masks(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Linear, nn.Conv2d)) and is_masked_module(m):
            if hasattr(m, 'fwd_hook'):
                m.fwd_hook.remove()
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Linear, nn.Conv2d)) and hasattr(m, 'weight_mask'):
            delattr(m, 'weight_mask')
            # m.register_buffer('weight_mask', torch.ones_like(m.weight.data).to(m.weight.device))
            if m.bias is not None and hasattr(m, 'bias_mask'):
                delattr(m, 'bias_mask')


def fill_model(model, bias=False):
    ## Remove ReLU inplace, Dropout ##
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0
        if isinstance(m, (nn.Dropout, nn.ReLU)):
            m.inplace = False

    ## Register masks ##
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if not hasattr(m, 'weight_mask'):
                print('Warning: model has no registered mask.')
                return
            # skip residual shortcuts
            m.weight.data.copy_(torch.where(m.weight_mask == 0.0, torch.tensor([1.]).to(m.weight.device), m.weight))
            if hasattr(m, 'bias_mask') and m.bias is not None and bias:
                m.bias.data.copy_(torch.where(m.bias_mask == 0.0, torch.tensor([1.]).to(m.bias.device), m.bias))

    print('Model weight has filled.')


def masked_params(model):
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            yield f'{name}.weight', module.weight_mask, module.weight
        if hasattr(module, 'bias_mask'):
            yield f'{name}.bias', module.bias_mask, module.bias


'''
    Only applicable for Layer.layers models
'''


def free_bias_special(model):
    bias_count = 0

    def _free_hooks(module):
        module.fwd_hook.remove()

    def _free_bias_masks(module):
        if hasattr(module, 'bias_mask'):
            delattr(module, 'bias_mask')

    for module in model.modules():
        if hasattr(module, 'bias_mask') and isinstance(module, (nn.Linear, nn.Conv2d)):
            _free_bias_masks(module)
            bias_count += 1
        if not hasattr(module, 'bias_mask') and not hasattr(module, 'weight_mask') and hasattr(module, 'fwd_hook'):
            _free_hooks(module)
    print(f'\tFree-bias ={bias_count} bias masks are deleted. ')


def free_modules(model, free_bn, free_Id, free_bias, free_conv1):
    def _free_weight_masks(module):
        if hasattr(module, 'weight_mask'):
            delattr(module, 'weight_mask')

    def _free_bias_masks(module):
        if hasattr(module, 'bias_mask'):
            delattr(module, 'bias_mask')

    def _free_hooks(module):
        module.fwd_hook.remove()

    is_conv1 = True
    bn_count, Id_count, bias_count = 0, 0, 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv2d)):
            if free_conv1 and is_conv1:
                _free_weight_masks(module)
                _free_bias_masks(module)
                is_conv1 = False
                bias_count += 1

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm2d)):
            if free_bn:
                _free_weight_masks(module)
                _free_bias_masks(module)
                bn_count += 1
                bias_count += 1

        if free_bias:
            if hasattr(module, 'bias_mask') and isinstance(module, (nn.Conv2d, nn.Conv2d)):
                _free_bias_masks(module)
                bias_count += 1

        if not hasattr(module, 'bias_mask') and not hasattr(module, 'weight_mask') and hasattr(module, 'fwd_hook'):
            _free_hooks(module)

    print(f'\n\tFree-bn = {free_bn}, {bn_count} bn masks are deleted.')
    print(f'\tFree-Id = {free_Id}, {Id_count} Identity masks are deleted. (always 0)')
    print(f'\tFree-bias = {free_bias}, {bias_count} bias masks are deleted. (FC-bias is excluded!)')
    print(f'\tFree-Conv1 = {free_conv1}.')


def _quantile_helper(x):
    masks_flatten = x.flatten().detach()
    violation_0 = masks_flatten.le(0.0).sum().item()
    violation_1 = masks_flatten.ge(1.0).sum().item()
    keep_0 = masks_flatten.eq(0.0).sum().item()
    return {'mle0': violation_0 / masks_flatten.numel() * 100,
            'mge1': violation_1 / masks_flatten.numel() * 100,
            'meq0': keep_0 / masks_flatten.numel() * 100,
            'qt1': torch.quantile(masks_flatten, 0.01).item(),
            'qt10': torch.quantile(masks_flatten, 0.1).item(),
            'qt50': torch.quantile(masks_flatten, 0.5).item(),
            'qt90': torch.quantile(masks_flatten, 0.9).item(),
            'qt99': torch.quantile(masks_flatten, 0.99).item()
            }


def quantile_helper_dict(x, theta, mask_dim=None):
    if isinstance(x, list):
        masks_flatten_x = torch.cat([xx.flatten().detach() for xx in x]).flatten()  # .cpu().numpy()
        masks_flatten_theta = torch.cat([xx.flatten().detach() for xx in theta]).flatten()  # .cpu().numpy()
        if mask_dim is None:
            masks_flatten_x_theta = torch.cat(
                [xx.flatten().detach() * yy.flatten().detach() for xx, yy in zip(x, theta)]).flatten()  # .cpu().numpy()
        else:
            masks_flatten_x_theta = []
            for xx, yy in zip(x, theta):

                if yy.dim() == 4:
                    view_shape = (-1, 1, 1, 1) if mask_dim == 0 else (1, -1, 1, 1)
                elif yy.dim() == 2:
                    view_shape = (-1, 1) if mask_dim == 0 else (1, -1)
                else:
                    view_shape = (-1)

                weight_mask_expanded = xx.view(view_shape).expand_as(yy)
                masks_flatten_x_theta.append((weight_mask_expanded * yy).flatten().detach())
            masks_flatten_x_theta = torch.cat(masks_flatten_x_theta).flatten()
    else:
        masks_flatten_x = x.flatten().detach()
        masks_flatten_theta = theta.flatten().detach()
        if mask_dim is None:
            masks_flatten_x_theta = masks_flatten_x * masks_flatten_theta
        else:
            masks_flatten_x_theta = []
            if theta.dim() == 4:
                view_shape = (-1, 1, 1, 1) if mask_dim == 0 else (1, -1, 1, 1)
            elif theta.dim() == 2:
                view_shape = (-1, 1) if mask_dim == 0 else (1, -1)
            else:
                view_shape = (-1)

            weight_mask_expanded = x.view(view_shape).expand_as(theta)
            masks_flatten_x_theta = (weight_mask_expanded * theta).flatten().detach()

    d_x = _quantile_helper(masks_flatten_x)
    d_x_theta = _quantile_helper(masks_flatten_x_theta)
    return {'mask': d_x, 'mask_theta': d_x_theta}


def classification_helper():
    en = nn.CrossEntropyLoss()

    def helper(model, x, y):
        return en(model(x), y)

    return helper


def transformers_helper():
    def helper(model, x, y):
        output = model(**x)
        return output.loss

    return helper
