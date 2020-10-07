import os
from munch import munchify
from utils import *
import torch.optim as optim
from trainer import *
from tqdm import tqdm_notebook as tqdm


def train_teacher(params):
    params = munchify(params)
    device = params.device

    # Result Dict
    result = {
        'org_acc': [],
        # 'adv_acc': [],
        'model': None,
    }

    # Setup Model
    model = get_models(params.network)
    model.to(device)

    # Setup Dataloader, Optimizer, Criterion
    train_loader, test_loader = get_dataloader(params.dataset, params.batch_size)
    optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, params.lr_steps)
    criterion = nn.CrossEntropyLoss()

    # Adversarial Training
    for epoch in tqdm(range(params.epochs)):
        train(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        scheduler.step(epoch)
        result['org_acc'].append(test(test_loader, model, device))
        # result['adv_acc'].append(complex_test(params.adv, test_loader, model, device))
        print('[', epoch, ']\t', result['org_acc'][-1])

    # Save Result
    result['model'] = model.to('cpu')
    if not os.path.exists(params.save_path):
        os.makedirs(params.save_path)
    torch.save(result, os.path.join(params.save_path, params.save_name))
