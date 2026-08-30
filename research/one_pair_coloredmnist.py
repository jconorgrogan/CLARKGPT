import argparse
import copy
import itertools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST

ENVIRONMENTS = [0.1, 0.2, 0.9]
TRAIN_VAL_SPLIT = 0.8
LABEL_NOISE = 0.25
LR = 1e-3


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bernoulli(p: float, n: int) -> torch.Tensor:
    return (torch.rand(n) < p).float()


def xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs()


def color_dataset(images: torch.Tensor, labels_0_9: torch.Tensor, env: float):
    y = (labels_0_9 < 5).float()
    y = xor(y, bernoulli(LABEL_NOISE, len(y)))
    color = xor(y, bernoulli(env, len(y)))
    x = torch.stack([images, images], dim=1)
    x[torch.arange(len(x)), (1 - color).long()] *= 0
    return x.float().div_(255.0), y.long()


def build_benchmark(seed: int, root: str):
    seed_all(seed)
    train = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)
    raw_x = torch.cat([train.data, test.data])
    raw_y = torch.cat([train.targets, test.targets])
    p = torch.randperm(len(raw_x))
    raw_x, raw_y = raw_x[p], raw_y[p]

    envs, raw_envs = [], []
    for i, e in enumerate(ENVIRONMENTS):
        xi, yi = raw_x[i::3], raw_y[i::3]
        envs.append(color_dataset(xi, yi, e))
        raw_envs.append((xi, yi))

    train_parts, val_parts, raw_train_parts = [], [], []
    for i in range(2):
        x, y = envs[i]
        xi, yi = raw_envs[i]
        n_train = math.ceil(len(x) * TRAIN_VAL_SPLIT)
        split_p = torch.randperm(len(x), generator=torch.Generator().manual_seed(42))
        ti, vi = split_p[:n_train], split_p[n_train:]
        train_parts.append((x[ti], y[ti]))
        val_parts.append((x[vi], y[vi]))
        raw_train_parts.append((xi[ti], yi[ti]))

    xtr = torch.cat([a for a, _ in train_parts])
    ytr = torch.cat([b for _, b in train_parts])
    xva = torch.cat([a for a, _ in val_parts])
    yva = torch.cat([b for _, b in val_parts])
    raw_train_x = torch.cat([a for a, _ in raw_train_parts]).float().div_(255.0)
    raw_train_y = torch.cat([b for _, b in raw_train_parts])
    xte, yte = envs[2]
    return xtr, ytr, xva, yva, xte, yte, raw_train_x, raw_train_y


def one_perfect_pair(raw_x: torch.Tensor, raw_y: torch.Tensor, seed: int):
    g = torch.Generator().manual_seed(10000 + seed)
    binary = (raw_y < 5).long()
    pool = torch.where(binary == 1)[0]
    idx = pool[torch.randperm(len(pool), generator=g)[0]]
    image = raw_x[idx]
    zero = torch.zeros_like(image)
    return torch.stack([image, zero]).unsqueeze(0), torch.stack([zero, image]).unsqueeze(0)


def infer_channel_permutation(left: torch.Tensor, right: torch.Tensor):
    scored = []
    for perm in itertools.permutations(range(left.shape[1])):
        p = list(perm)
        residual = ((left[:, p] - right) ** 2).mean() + ((right[:, p] - left) ** 2).mean()
        scored.append((float(residual), p))
    scored.sort(key=lambda z: z[0])
    return scored[0][1], scored


def quotient_project(x: torch.Tensor, perm):
    return x + x[:, perm]


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.gn0 = nn.GroupNorm(8, 64)
        self.gn1 = nn.GroupNorm(8, 128)
        self.gn2 = nn.GroupNorm(8, 128)
        self.gn3 = nn.GroupNorm(8, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(128, 2, bias=True)

    def forward(self, x):
        x = self.gn0(F.relu(self.conv1(x)))
        x = self.gn1(F.relu(self.conv2(x)))
        x = self.gn2(F.relu(self.conv3(x)))
        x = self.gn3(F.relu(self.conv4(x)))
        return self.head(self.pool(x).reshape(len(x), -1))


@torch.no_grad()
def accuracy(model, x, y, batch_size: int):
    model.eval()
    correct = 0
    for start in range(0, len(x), batch_size):
        pred = model(x[start:start + batch_size]).argmax(1)
        correct += int((pred == y[start:start + batch_size]).sum())
    return correct / len(y)


def train(seed: int, epochs: int, batch_size: int, root: str):
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.backends.mkldnn.enabled = True
    xtr, ytr, xva, yva, xte, yte, raw_x, raw_y = build_benchmark(seed, root)
    left, right = one_perfect_pair(raw_x, raw_y, seed)
    perm, scores = infer_channel_permutation(left, right)

    xtr = quotient_project(xtr, perm)
    xva = quotient_project(xva, perm)
    xte = quotient_project(xte, perm)

    seed_all(seed)
    model = MNISTCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    order_rng = torch.Generator().manual_seed(20000 + seed)
    best_val = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xtr), generator=order_rng)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()
        val = accuracy(model, xva, yva, 2048)
        print(json.dumps({'epoch': epoch + 1, 'val': val}), flush=True)
        if val > best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    pair_error = float(((quotient_project(left, perm) - quotient_project(right, perm)) ** 2).mean())
    return {
        'seed': seed,
        'epochs': epochs,
        'batch_size': batch_size,
        'pair_budget': 1,
        'learned_permutation': perm,
        'candidate_residuals': [{'residual': r, 'perm': p} for r, p in scores],
        'pair_projection_error': pair_error,
        'best_train_env_val_acc': best_val,
        'test_acc': accuracy(model, xte, yte, 2048),
        'parameters': sum(p.numel() for p in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--root', default='mnist_data')
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    row = train(args.seed, args.epochs, args.batch_size, args.root)
    payload = {'seconds': time.time() - t0, 'row': row}
    out = Path(args.out) / f'seed_{args.seed}.json'
    out.write_text(json.dumps(payload, indent=2))
    print('RESULT', json.dumps(row, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
