import os
import random

import torch

root_dir = "../../testing/testing/MNIST/"

orig = torch.load(root_dir + 'original/processed/training.pt')
orig_test = torch.load(root_dir + 'original/processed/test.pt')


def create_imbalanced(dataset, proportions: dict, save_path: str, trng_or_test: str):
    new_ims = list()
    new_labs = list()
    for i in range(len(dataset[0])):
        if proportions[int(dataset[1][i])] > 0:
            new_ims.append(dataset[0][i])
            new_labs.append(dataset[1][i])
            proportions[int(dataset[1][i])] -= 1

    if not os.path.exists(root_dir + save_path):
        print("making dirs")
        os.makedirs(root_dir + save_path)

    torch.save((torch.stack(new_ims), torch.stack(new_labs)), root_dir + save_path +
               trng_or_test)


def create_corrupted(dataset, p: float, save_path: str, trng_or_test: str):
    new_labs = list()
    count = 0
    for i in range(len(dataset[0])):
        if random.random() < p:
            count += 1
            new_labs.append(torch.tensor(random.randint(0, 9)))
        else:
            new_labs.append(dataset[1][i])
    print("{} labels changed out of {}".format(count, len(dataset[0])))
    torch.save((dataset[0], torch.stack(new_labs)), root_dir + save_path + trng_or_test)


if __name__ == '__main__':
    pass
    # props = {0: 445, 1: 444, 2: 444, 3: 444, 4: 444, 5: 1000, 6: 444, 7: 445, 8: 445, 9: 445}
    # create_imbalanced(orig, proportions=props, save_path="imbalanced/processed/",
    #                   trng_or_test="training.pt")

    # props = {0: 511, 1: 511, 2: 511, 3: 511, 4: 511, 5: 400, 6: 511, 7: 511, 8: 511, 9: 512}
    # create_imbalanced(orig, proportions=props, save_path="imbalanced_low/processed/",
    #                   trng_or_test="training.pt")
    #
    props = {0: 500, 1: 500, 2: 500, 3: 500, 4: 500, 5: 500, 6: 500, 7: 500, 8: 500, 9: 500}
    create_imbalanced(orig, proportions=props, save_path="corrupted/processed/",
                      trng_or_test="training.pt")

    d = torch.load(root_dir + 'corrupted/processed/training.pt')
    create_corrupted(dataset=d, p=0.1, save_path="corrupted/processed/",
                     trng_or_test="training.pt")
