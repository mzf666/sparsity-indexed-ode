import os

import torch

from Methods import utils
from Utils import datasets
from Utils import routine


def load_data(root, dataset, batch_size, num_workers, imsize=None):
    print(f'Loading {dataset} dataset.')
    train_loader = datasets.dataloader(root, dataset, batch_size, True, workers=num_workers, imsize=imsize)
    test_loader = datasets.dataloader(root, dataset, batch_size, False, workers=num_workers, imsize=imsize)

    return train_loader, test_loader


def load_model(root, model_type, dataset, device, pretrained=True, load_checkpoint=None):
    from Models import mlp, lottery_vgg, lottery_resnet, tinyimagenet_vgg, tinyimagenet_resnet, imagenet_vgg, \
        imagenet_resnet
    default_models = {
        'fc': mlp.fc,
        'conv': mlp.conv,
    }
    lottery_models = {
        'vgg11': lottery_vgg.vgg11,
        'vgg11_bn': lottery_vgg.vgg11_bn,
        'vgg13': lottery_vgg.vgg13,
        'vgg13_bn': lottery_vgg.vgg13_bn,
        'vgg16': lottery_vgg.vgg16,
        'vgg16_bn': lottery_vgg.vgg16_bn,
        'vgg19': lottery_vgg.vgg19,
        'vgg19_bn': lottery_vgg.vgg19_bn,
        'resnet20': lottery_resnet.resnet20,
        'resnet32': lottery_resnet.resnet32,
        'resnet44': lottery_resnet.resnet44,
        'resnet56': lottery_resnet.resnet56,
        'resnet110': lottery_resnet.resnet110,
        'resnet1202': lottery_resnet.resnet1202,
        'wrn20': lottery_resnet.wide_resnet20,
        'wrn32': lottery_resnet.wide_resnet32,
        'wrn44': lottery_resnet.wide_resnet44,
        'wrn56': lottery_resnet.wide_resnet56,
        'wrn110': lottery_resnet.wide_resnet110,
        'wrn1202': lottery_resnet.wide_resnet1202,
        'mobilenet_v2': lottery_resnet.mobilenet_v2,
    }
    tinyimagenet_models = {
        'vgg11': tinyimagenet_vgg.vgg11,
        'vgg11_bn': tinyimagenet_vgg.vgg11_bn,
        'vgg13': tinyimagenet_vgg.vgg13,
        'vgg13_bn': tinyimagenet_vgg.vgg13_bn,
        'vgg16': tinyimagenet_vgg.vgg16,
        'vgg16_bn': tinyimagenet_vgg.vgg16_bn,
        'vgg19': tinyimagenet_vgg.vgg19,
        'vgg19_bn': tinyimagenet_vgg.vgg19_bn,
        'resnet18': tinyimagenet_resnet.resnet18,
        'resnet34': tinyimagenet_resnet.resnet34,
        'resnet50': tinyimagenet_resnet.resnet50,
        'resnet101': tinyimagenet_resnet.resnet101,
        'resnet152': tinyimagenet_resnet.resnet152,
        'wrn18': tinyimagenet_resnet.wide_resnet18,
        'wrn34': tinyimagenet_resnet.wide_resnet34,
        'wrn50': tinyimagenet_resnet.wide_resnet50,
        'wrn101': tinyimagenet_resnet.wide_resnet101,
        'wrn152': tinyimagenet_resnet.wide_resnet152,
    }
    imagenet_models = {
        'vgg11': imagenet_vgg.vgg11,
        'vgg11_bn': imagenet_vgg.vgg11_bn,
        'vgg13': imagenet_vgg.vgg13,
        'vgg13_bn': imagenet_vgg.vgg13_bn,
        'vgg16': imagenet_vgg.vgg16,
        'vgg16_bn': imagenet_vgg.vgg16_bn,
        'vgg19': imagenet_vgg.vgg19,
        'vgg19_bn': imagenet_vgg.vgg19_bn,
        'resnet18': imagenet_resnet.resnet18,
        'resnet34': imagenet_resnet.resnet34,
        'resnet50': imagenet_resnet.resnet50,
        'resnet101': imagenet_resnet.resnet101,
        'resnet152': imagenet_resnet.resnet152,
        'wrn50': imagenet_resnet.wide_resnet50_2,
        'wrn101': imagenet_resnet.wide_resnet101_2,
    }
    models = {
        'default': default_models,
        'lottery': lottery_models,
        'tiny_imagenet': tinyimagenet_models,
        'imagenet': imagenet_models,
    }

    model_class = 'lottery' if dataset in ['cifar10', 'cifar100'] else dataset

    if dataset == 'imagenet':
        print("WARNING: ImageNet models do not implement `dense_classifier`.")

    model_kwargs = dict(
        cifar10={
            'input_shape': (3, 32, 32),
            'num_classes': 10,
        },
        cifar100={
            'input_shape': (3, 32, 32),
            'num_classes': 100,
        },
        tiny_imagenet={
            'input_shape': (3, 64, 64),
            'num_classes': 200,
        },
        imagenet={
            'input_shape': (3, 224, 224),
            'num_classes': 1000,
            # 'pretrained': True,
        },

    )

    model = models[model_class][model_type](**model_kwargs[dataset]).to(device)

    if pretrained and dataset != 'imagenet':
        ckpt_path = f'{root}/pretrained/{dataset}/{model_type}.pt'
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            print({k: ckpt[k] for k in ['epoch', 'acc']})
            ckpt_no_mask = {k: ckpt['model'][k] for k in ckpt['model'].keys() if 'mask' not in k}
            # model.load_state_dict(ckpt['model'], strict=True)
            model.load_state_dict(ckpt_no_mask, strict=True)
            print(f'\n\tPretrained {dataset} {model_type} loaded successfully.\n')
        else:
            if dataset != 'tiny_imagenet':
                print(f'\n\tWarning: NO avaible checkpoint in {ckpt_path}, training from scratch ...\n')
            else:
                import torchvision.models as imagenet_models
                print(f'\n\tWarning: NO avaible checkpoint in {ckpt_path}, training from scratch ...\n')
                if hasattr(imagenet_models, model_type):
                    imagenet_ckpt = getattr(imagenet_models, model_type)(pretrained=True).state_dict()
                    imagenet_ckpt = {k: imagenet_ckpt[k]
                                     for k in imagenet_ckpt.keys()
                                     if 'fc' not in k and 'classifier' not in k}
                    model.load_state_dict(imagenet_ckpt, strict=False)
                    print('\n\tWarning: loading ImageNet pretrained model (except FC) instead ...\n')

    if pretrained and dataset == 'imagenet':
        from torchvision.models import resnet50 as imresnet50
        from torchvision.models import vgg16_bn as imvgg16_bn  # , ResNet50_Weights, vgg16_bn
        models_image = {
            'resnet50': imresnet50,
            'vgg16_bn': imvgg16_bn,
        }
        models_checkpoint = {
            'resnet50': "IMAGENET1K_V1",
            'vgg16_bn': "IMAGENET1K_V1",
        }
        ckpt_path = f'{root}/pretrained/{dataset}/{model_type}.pt'

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            print({k: ckpt[k] for k in ckpt.keys() if k != 'state_dict'})
            pretrained_model = ckpt["state_dict"]
            model_state_dict = model.state_dict()

            pretrained_final = {
                k.replace("module.", ""): v
                for k, v in pretrained_model.items()
                if (k.replace("module.", "") in model_state_dict and v.size() == model_state_dict[
                    k.replace("module.", "")].size())
            }
            pretrained_final['fc.weight'] = pretrained_model['module.fc.weight'].squeeze(-1).squeeze(-1)
            model.load_state_dict(pretrained_final, strict=False)
            print(f'\nPretrained {dataset} {model_type} loaded successfully.\n')
        else:
            model = models_image[model_type](weights=models_checkpoint[model_type]).to(device)
            print(f'\nPretrained {dataset} {model_type} loaded from torchvision.\n')

    if load_checkpoint is not None:
        ckpt_path = load_checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        print(f'\nLoaded {dataset} {model_type} from a specific checkpoint ...\n')
    return model


def load_optimizer(opt):
    import torch.optim as optim
    optimizers = {
        'adam': (optim.Adam, {'weight_decay': 5e-5}),
        'sgd': (optim.SGD, {'weight_decay': 5e-5}),
        'momentum': (optim.SGD, {'momentum': 0.9, 'nesterov': True}),
        'rms': (optim.RMSprop, {}),
        'adamw': (optim.AdamW, {}),
    }
    return optimizers[opt]


def load_pruner(model, pruner, structural, mask_dim, **kwargs):
    print(f'pruner = {pruner}')
    if structural:
        from Methods import structural_pruners
        pruner = vars(structural_pruners)[pruner](model, mask_dim, **kwargs)
        print(f'\t Structural pruner loaded successfully, mask-dim = {mask_dim} ...')
    else:
        from Methods import pruners
        pruner = vars(pruners)[pruner](model, **kwargs)
        print('\t Unstructural pruner loaded successfully ...')

    return pruner


def load_prune_data_config(data, pruner):
    if pruner == 'SNIP':
        configs = {
            'cifar10': (100, 256),
            'cifar100': (1000, 256),
            'tiny_imagenet': (2000, 64),
            'imagenet': (10000, 16),
        }
    elif pruner == 'GraSP':
        configs = {
            'cifar10': (100, 256),
            'cifar100': (1000, 256),
            'tiny_imagenet': (2000, 64),
            'imagenet': (10000, 16),
        }
    elif pruner in ['ODE']:
        configs = {
            'cifar10': (100, 256),
            'cifar100': (5000, 256),
            'tiny_imagenet': (2000, 64),
            'imagenet': (10000, 16),
        }
    elif pruner in ['REG']:
        configs = {
            'cifar10': (100, 256),
            'cifar100': (5000, 256),
            'tiny_imagenet': (2000, 64),
            'imagenet': (10000, 16),
        }
    elif pruner in ['Rand', 'Mag', 'MagRand', 'SynFlow']:
        configs = {
            'cifar10': (1, 1),
            'cifar100': (1, 1),
            'tiny_imagenet': (1, 1),
            'imagenet': (1, 1),
        }
    else:
        raise NotImplementedError
    return configs[data]  # data size & batch size


## Only apply to masked-model
def load_sparsity(model, sparsity, scope='global'):
    if scope == 'global':
        return sparsity
    elif scope == 'local':
        return [sparsity for _, _, _ in utils.masked_params(model)]
    else:
        raise NotImplementedError


def load_sparsity_schedule(sparsity, epochs, schedule='exponential', end_sparsity=None):
    def _get_sparse(sparsity, epoch, schedule):
        if schedule == 'exponential':
            return sparsity ** ((epoch + 1) / epochs)
        elif schedule == 'linear':
            return 1.0 - (1.0 - sparsity) * ((epoch + 1) / epochs)
        elif schedule == 'increase':
            if end_sparsity:
                assert sparsity < end_sparsity <= 1.0
                end_s = end_sparsity
            else:
                end_s = 1.0
            return sparsity + (end_s - sparsity) * ((epoch + 1) / epochs)

    if isinstance(sparsity, list):
        return [[_get_sparse(s, epoch, schedule)
                 for s in sparsity]
                for epoch in range(epochs)]
    else:
        return [_get_sparse(sparsity, epoch, schedule)
                for epoch in range(epochs)]


def load_train_function(dataset, model_name):
    if dataset in ['cifar10', 'cifar100', 'tiny_imagenet']:
        train_func = routine.train
    elif dataset in ['imagenet']:
        train_func = routine.train_image
    else:
        raise NotImplementedError
    return train_func


def load_evaluate_function(dataset: str, model_name, data_dir):
    if dataset in ['cifar10', 'cifar100', 'tiny_imagenet']:
        def eval(model, data_loader, device):
            import torch
            return routine.evaluate(model, torch.nn.CrossEntropyLoss(), data_loader, device)
    elif dataset in ['imagenet']:
        def eval(model, data_loader, device):
            import torch
            return routine.evaluate(model, torch.nn.CrossEntropyLoss(), data_loader, device)
    else:
        raise NotImplementedError
    return eval
