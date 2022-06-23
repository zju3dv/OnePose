import torch
import os
from collections import OrderedDict


def load_model(net, optim, scheduler, recorder, model_dir, resume=True, epoch=-1):
    if not resume:
        os.system('rm -rf {}'.format(model_dir))

    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('Load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1


def save_model(net, optim, scheduler, recorder, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) <= 200:
        return
    os.system('rm {}'.format(os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def load_network_ckpt(net, ckpt_path):
    pretrained_model = torch.load(ckpt_path, torch.device('cpu'))
    pretrained_model = pretrained_model['state_dict']

    pretrained_model = remove_net_layer(pretrained_model, 'detector')
    pretrained_model = remove_net_prefix(pretrained_model, 'superglue.')

    print('=> load weights: ', ckpt_path)
    net.load_state_dict(pretrained_model)
    return None


def load_network(net, model_dir, resume=True, epoch=-1, strict=True, force=False):
    """
    Load latest network-weights from dir or path
    """
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        if force:
            raise NotImplementedError
        else:
            print('pretrained model does not exist')
            return 0

    if os.path.isdir(model_dir):
        pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if 'pth' in pth]
        if len(pths) == 0:
            return 0
        if epoch == -1:
            pth = max(pths)
        else:
            pth = epoch
        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    print('=> load weights: ', model_path)
    pretrained_model = torch.load(model_path, torch.device("cpu"))
    if 'net' in pretrained_model.keys():
        net.load_state_dict(pretrained_model['net'], strict=strict)
    else:
        net.load_state_dict(pretrained_model, strict=strict)
    return pretrained_model.get('epoch', 0) + 1


def remove_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(prefix):
            net_[k[len(prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def add_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        net_[prefix + k] = net[k]
    return net_


def replace_net_prefix(net, orig_prefix, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(orig_prefix):
            net_[prefix + k[len(orig_prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def remove_net_layer(net, layers):
    keys = list(net.keys())
    for k in keys:
        for layer in layers:
            if k.startswith(layer):
                del net[k]
    return net


def to_cuda(data):
    if type(data).__name__ == "Tensor":
        data = data.cuda()
    elif type(data).__name__ == 'list':
        data = [d.cuda() for d in data]
    elif type(data).__name__ == 'dict':
        data = {k: v.cuda() for k, v in data.items()}
    else:
        raise NotImplementedError
    return data
