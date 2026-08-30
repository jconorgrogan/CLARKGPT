import argparse
import copy
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bernoulli(p: float, n: int) -> torch.Tensor:
    return (torch.rand(n) < p).float()


def xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs()


def color_environment(images: torch.Tensor, labels: torch.Tensor, env: float):
    y = (labels < 5).float()
    y = xor(y, bernoulli(0.25, len(y)))
    colors = xor(y, bernoulli(env, len(y)))
    x = torch.stack([images, images], dim=1)
    x[torch.arange(len(x)), (1 - colors).long()] *= 0
    return x.float().div_(255.0), y.long()


def split_environment(x, y):
    n_train = math.ceil(len(x) * 0.8)
    g = torch.Generator().manual_seed(42)
    p = torch.randperm(len(x), generator=g)
    return (x[p[:n_train]], y[p[:n_train]]), (x[p[n_train:]], y[p[n_train:]])


def build_data(seed: int, root: str):
    seed_all(seed)
    tr = MNIST(root, train=True, download=True)
    te = MNIST(root, train=False, download=True)
    raw_x = torch.cat([tr.data, te.data])
    raw_y = torch.cat([tr.targets, te.targets])
    p = torch.randperm(len(raw_x))
    raw_x, raw_y = raw_x[p], raw_y[p]

    envs, raw_envs = [], []
    for i, e in enumerate([0.1, 0.2, 0.9]):
        xi, yi = raw_x[i::3], raw_y[i::3]
        envs.append(color_environment(xi, yi, e))
        raw_envs.append((xi, yi))

    tr_parts, va_parts, pair_x = [], [], []
    for i in range(2):
        (xa, ya), (xv, yv) = split_environment(*envs[i])
        tr_parts.append((xa, ya))
        va_parts.append((xv, yv))
        n_train = len(xa)
        idx = torch.randperm(len(raw_envs[i][0]), generator=torch.Generator().manual_seed(42))[:n_train]
        pair_x.append(raw_envs[i][0][idx])

    xtr = torch.cat([a for a, _ in tr_parts])
    ytr = torch.cat([b for _, b in tr_parts])
    xva = torch.cat([a for a, _ in va_parts])
    yva = torch.cat([b for _, b in va_parts])
    xte, yte = envs[2]
    pair_x = torch.cat(pair_x).float().div_(255.0)
    return xtr, ytr, xva, yva, xte, yte, pair_x


def infer_permutation(one_raw_image: torch.Tensor):
    z = torch.zeros_like(one_raw_image)
    left = torch.stack([one_raw_image, z])
    right = torch.stack([z, one_raw_image])
    scored = []
    for p in itertools.permutations(range(2)):
        pp = list(p)
        loss = ((left[pp] - right) ** 2).mean() + ((right[pp] - left) ** 2).mean()
        scored.append((float(loss), pp))
    scored.sort(key=lambda z: z[0])
    return scored[0][1], scored[0][0], scored[1][0]


def project(x: torch.Tensor, perm):
    return x + x[:, perm]


class SmallLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def accuracy(model, x, y, batch=2048):
    model.eval()
    correct = 0
    for a in range(0, len(x), batch):
        pred = model(x[a:a+batch]).argmax(1)
        correct += int((pred == y[a:a+batch]).sum())
    return correct / len(x)


def train(seed: int, method: str, epochs: int, root: str):
    torch.set_num_threads(4)
    xtr, ytr, xva, yva, xte, yte, pair_x = build_data(seed, root)
    perm, best_resid, second_resid = infer_permutation(pair_x[0])
    if method == 'compiled_1pair':
        xtr, xva, xte = project(xtr, perm), project(xva, perm), project(xte, perm)

    seed_all(seed)
    model = SmallLeNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state = None
    best_val = -1.0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xtr))
        for a in range(0, len(order), 1024):
            ix = order[a:a+1024]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xtr[ix]), ytr[ix])
            loss.backward()
            opt.step()
        val = accuracy(model, xva, yva)
        if val > best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {
        'seed': seed,
        'method': method,
        'epochs': epochs,
        'parameters': sum(p.numel() for p in model.parameters()),
        'pair_budget': 1 if method == 'compiled_1pair' else 0,
        'learned_permutation': perm,
        'best_pair_residual': best_resid,
        'second_pair_residual': second_resid,
        'best_train_env_val_acc': best_val,
        'test_acc': accuracy(model, xte, yte),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--root', default='mnist_data')
    p.add_argument('--out', default='results')
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = [train(args.seed, m, args.epochs, args.root) for m in ['erm', 'compiled_1pair']]
    payload = {'seconds': time.time() - t0, 'rows': rows}
    out = Path(args.out) / f'seed_{args.seed}.json'
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
