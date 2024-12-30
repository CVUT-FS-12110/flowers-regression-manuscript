import time
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from glob import glob
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold

# from nn import MultiViewNetwork # TODO Choose this or the other one
from nn_fpn import MultiViewNetwork

from reference_nn import ReferenceMultiViewNetwork
from dataloader import CustomDataset, set_random_seed



def train(model, criterion, device, optimizer, epoch, train_loader):
    model.train()


    for batch_idx, (images_a, images_b, targets, names) in enumerate(train_loader):
        images_a, images_b, targets = images_a.to(device), images_b.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_a, images_b)#.squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        log_interval = 10
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_a), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_error = 0

    with torch.no_grad():
        for batch_idx, (images_a, images_b, targets, names) in enumerate(test_loader):
            images_a, images_b, targets = images_a.to(device), images_b.to(device), targets.to(device)
            outputs = model(images_a, images_b).squeeze()

            pred = np.array(outputs.cpu()).copy()
            truth = np.array(targets.cpu()).copy().T[0]
            dif = truth - pred
            test_error += np.mean(np.abs(dif))

    print()
    print("Validations:", test_error)
    print(truth.astype("int"))
    print(dif.round().astype("int"))
    return test_error, dif, names, pred


def main(seed):
    folds = 10
    epochs = 300
    ground_truth = False #  visually-estimated / field-validated
    use_cuda = True
    batch_size = 8
    reference_model = False


    set_random_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': batch_size, "shuffle": True}
    test_kwargs = {'batch_size': 16, "shuffle": False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)




    data_path = "final_data/*.jpg"
    filenames = list(glob(data_path))

    observations = sorted(list(set([tuple(os.path.basename(name).split(".")[0].split("_")[0:3]) for name in filenames])))
    trees = sorted(list(set([observation[1:3] for observation in observations])))

    np.random.shuffle(trees)


    kfold = KFold(n_splits=folds, shuffle=False)

    results = []
    convergence = np.zeros([epochs, folds])

    for fold, (train_ids, test_ids) in enumerate(kfold.split(trees)):
        print()
        print("Fold:", fold + 1)

        train_trees = [trees[i] for i in train_ids]
        test_trees = [trees[i] for i in test_ids]

        print(test_trees)

        train_samples = [observation for observation in observations if observation[1:3] in train_trees]
        test_samples = [observation for observation in observations if observation[1:3] in test_trees]

        train_dataset = CustomDataset(train_samples, data_path, ground_truth=ground_truth)
        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

        test_dataset = CustomDataset(test_samples, data_path, augment=False, ground_truth=ground_truth)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        if not reference_model:
            print("Proposed method")
            model = MultiViewNetwork().to(device)
            criterion = nn.SmoothL1Loss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) # TODO switch
        else:
            print("Reference method (CountNet)")
            model = ReferenceMultiViewNetwork().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.972351)

        best_error = 1000
        best_results = None
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train(model, criterion, device, optimizer, epoch, train_loader)
            t1 = time.time() - t0
            scheduler.step()
            test_error, test_errors, names, predictions = test(model, device, test_loader)
            convergence[epoch - 1, fold] = test_error
            if test_error < best_error:
                best_error = test_error
                best_results = [(str(name), str(pred), str(error), str(epoch)) for name, error, pred in zip(names, list(test_errors), list(predictions))]
                torch.save(model.state_dict(), f"checkpoints/model_{seed}_{fold + 1}.pt")

            with open(f"results/execution.txt", "a") as f:
                f.write(str(t1) + "\n")

        results += best_results

    out_data = "\n".join(["\t".join(line) for line in results])
    with open(f"results/results_{seed}_{ground_truth}.tsv", "w") as f:
        f.write(out_data)
    np.savetxt(f"results/convergence_{seed}_{ground_truth}.tsv", convergence, delimiter="\t")


if __name__ == '__main__':
    for seed in [1, 12, 23, 34, 45, 56, 67, 78, 89, 90]: # seeds chosen according to pattern
        main(seed)



