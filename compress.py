import os
from munch import munchify
from utils import *
import torch.optim as optim
from trainer import *
from tqdm import tqdm_notebook as tqdm
from warmup_scheduler import GradualWarmupScheduler
import random
from prun import *


# def compress(device, config_path='configs/compress_config.yaml'):
def compress(params):

    params = munchify(params)
    device = params.general_params.device

    # Result Dict
    result = {
        'params': params,
        'initial_org_acc': 0.0,
        'initial_adv_acc': 0.0,
        'proj_org_acc': [],
        'proj_adv_acc': [],
        'model': None,
    }

    # Input Params
    general = params.general_params
    projection = params.projection_params
    adversarial = params.adversarial_params
    distillation = params.distillation_params

    # Fix Random Seed
    torch.manual_seed(general.random_seed)
    torch.cuda.manual_seed(general.random_seed)
    np.random.seed(general.random_seed)
    random.seed(general.random_seed)
    torch.backends.cudnn.deterministic = True

    # Setup Model(Pretrained)
    old_model = None
    teacher = None
    if general.pretrained_path:
        old_model = (torch.load(general.pretrained_path)['model'])
    else:
        old_model = get_models(general.network)

    assert old_model
    old_model.to(device)

    if distillation:
        teacher = (torch.load(distillation.teacher_path)['model'])
        teacher.to(device)

    else:
        teacher = None

    # Setup Dataloader, Optimizer, Criterion
    train_loader, test_loader = get_dataloader(general.dataset, seed=general.random_seed)
    if general.optimizer == 'sgd':
        optimizer = optim.SGD(old_model.parameters(), lr=general.lr_init, momentum=0.9, weight_decay=general.weight_decay)
    elif general.optimizer == 'adam':
        optimizer = optim.Adam(old_model.parameters(), lr=general.lr_init, weight_decay=general.weight_decay)
    else:
        assert False

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, general.lr_steps, general.lr_decay)
    if general.warm:
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    criterion = nn.CrossEntropyLoss()

    # Initial Performance
    result['initial_org_acc'] = test(test_loader, old_model, device)
    result['initial_adv_acc'] = complex_test(params, test_loader, old_model, device)

    print('[', 'init', ']\t',
          result['initial_org_acc'],
          result['initial_adv_acc'],
          (result['initial_org_acc'] + result['initial_adv_acc']) / 2,
          sparsity(old_model)
          )

    # Train & Projection
    for epoch in tqdm(range(general.epoch)):
        if general.warm:
            scheduler_warmup.step(epoch)
        else:
            scheduler.step(epoch)
        complex_train(
            params=params,
            teacher=teacher,
            loader=train_loader,
            model=old_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        result['proj_org_acc'].append(test(test_loader, old_model, device))
        result['proj_adv_acc'].append(complex_test(params, test_loader, old_model, device))
        print('[', epoch, ']\t',
              result['proj_org_acc'][-1],
              result['proj_adv_acc'][-1],
              (result['proj_org_acc'][-1]+result['proj_adv_acc'][-1])/2,
              sparsity(old_model)
              )

    # Save Result
    result['model'] = old_model.to('cpu')
    torch.save(result, os.path.join(general.save_path, general.save_name))

#
# if __name__ == "__main__":
#     compress()
