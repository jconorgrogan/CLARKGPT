import json, os, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

OUT = Path(os.environ.get("OUT", "causalworld_results"))
OUT.mkdir(parents=True, exist_ok=True)


def make_env(goal_x, seed):
    from causal_world.envs import CausalWorld
    from causal_world.task_generators import generate_task
    goals = {
        "default_goal_60": np.array([goal_x, 0.00, 0.10]),
        "default_goal_120": np.array([goal_x, 0.00, 0.13]),
        "default_goal_300": np.array([goal_x, 0.00, 0.16]),
    }
    task = generate_task(task_generator_id="reaching", **goals)
    return CausalWorld(
        task=task,
        skip_frame=20,
        enable_visualization=False,
        seed=seed,
        observation_mode="pixel",
        normalize_observations=False,
        max_episode_length=20,
        camera_indicies=np.array([0]),
    )


def set_color(env, color):
    success, obs = env.do_intervention({"stage_color": np.asarray(color, dtype=float)})
    if success is False:
        raise RuntimeError("stage_color intervention rejected")
    return np.asarray(obs)


def image_tensor(obs):
    x = np.asarray(obs)
    # Pixel mode with one camera normally returns [current, goal, H, W, C].
    if x.ndim == 4:
        pass
    elif x.ndim == 3:
        x = x[None, ...]
    else:
        raise RuntimeError(f"Unexpected pixel observation shape: {x.shape}")
    if x.shape[-1] not in (3, 4):
        raise RuntimeError(f"Expected RGB(A) final dimension: {x.shape}")
    x = x[..., :3].astype(np.float32)
    if x.max() > 1.5:
        x /= 255.0
    # Fast deterministic downsample to <=32x32.
    h, w = x.shape[-3], x.shape[-2]
    sh, sw = max(1, h // 32), max(1, w // 32)
    x = x[:, ::sh, ::sw, :][:, :32, :32, :]
    return x


def collect(envs, n, aligned, seed, random_color=False):
    rng = np.random.RandomState(seed)
    images, labels = [], []
    c0 = np.array([0.10, 0.22, 0.85])
    c1 = np.array([0.85, 0.18, 0.10])
    for _ in range(n):
        y = int(rng.randint(0, 2))
        env = envs[y]
        env.reset()
        # Vary the foreground robot pose while preserving the goal label.
        for _ in range(int(rng.randint(0, 3))):
            env.step(env.action_space.sample())
        if random_color:
            bit = int(rng.randint(0, 2))
        else:
            copy = rng.rand() < 0.95
            bit = y if copy else 1 - y
            if not aligned:
                bit = 1 - bit
        images.append(image_tensor(set_color(env, c1 if bit else c0)))
        labels.append(y)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def collect_pairs(envs, pairs, seed):
    rng = np.random.RandomState(seed)
    c0 = np.array([0.10, 0.22, 0.85])
    c1 = np.array([0.85, 0.18, 0.10])
    out = []
    for _ in range(pairs):
        y = int(rng.randint(0, 2))
        env = envs[y]
        env.reset()
        for _ in range(int(rng.randint(0, 3))):
            env.step(env.action_space.sample())
        a = image_tensor(set_color(env, c0))
        b = image_tensor(set_color(env, c1))
        out.append((a, b))
    return out


def compile_mask(pairs):
    diff = np.mean([np.abs(a - b) for a, b in pairs], axis=0)
    score = diff.mean(axis=-1, keepdims=True)
    positive = score[score > 1e-7]
    threshold = max(0.02, float(np.quantile(positive, 0.15))) if positive.size else 1.0
    mask = score >= threshold
    return np.repeat(mask, 3, axis=-1), score


def transform(X, mask=None):
    Z = X.copy()
    if mask is not None:
        Z[:, mask] = 0.5
    return Z.reshape(len(Z), -1)


def fit_curve(X, y, Xid, yid, Xood, yood, seed, epochs=24):
    rng = np.random.RandomState(8000 + seed)
    clf = SGDClassifier(
        loss="log_loss", alpha=3e-5, learning_rate="constant", eta0=0.03,
        penalty="l2", random_state=seed, average=True,
    )
    rows = []
    classes = np.array([0, 1])
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y))
        clf.partial_fit(X[order], y[order], classes=classes)
        rows.append({
            "epoch": epoch,
            "id_accuracy": accuracy_score(yid, clf.predict(Xid)),
            "ood_accuracy": accuracy_score(yood, clf.predict(Xood)),
        })
    return rows


def first_epoch(rows, threshold):
    for r in rows:
        if r["ood_accuracy"] >= threshold:
            return r["epoch"]
    return np.nan


def main():
    t0 = time.time()
    envs = [make_env(-0.045, 0), make_env(0.045, 1)]
    try:
        # One shared rendered dataset; model uncertainty comes from optimization seeds.
        Xtr, ytr = collect(envs, 800, aligned=True, seed=11)
        Xid, yid = collect(envs, 400, aligned=True, seed=12)
        Xood, yood = collect(envs, 800, aligned=False, seed=13)
        Xrand, yrand = collect(envs, 800, aligned=True, seed=14, random_color=True)
        pairs4 = collect_pairs(envs, 4, seed=21)
        pairs64 = collect_pairs(envs, 64, seed=22)
    finally:
        for env in envs:
            env.close()

    mask4, score4 = compile_mask(pairs4)
    mask64, _ = compile_mask(pairs64)
    rng = np.random.RandomState(99)
    random_mask = np.zeros(mask4.shape, dtype=bool)
    flat = random_mask.reshape(-1)
    flat[rng.choice(len(flat), size=int(mask4.sum()), replace=False)] = True
    random_mask = flat.reshape(mask4.shape)

    methods = {
        "raw_erm": (transform(Xtr), transform(Xid), transform(Xood)),
        "four_pair_compiler": (transform(Xtr, mask4), transform(Xid, mask4), transform(Xood, mask4)),
        "sixtyfour_pair_oracle": (transform(Xtr, mask64), transform(Xid, mask64), transform(Xood, mask64)),
        "random_mask_control": (transform(Xtr, random_mask), transform(Xid, random_mask), transform(Xood, random_mask)),
        "domain_randomized": (transform(Xrand), transform(Xid), transform(Xood)),
    }

    run_rows = []
    for seed in range(8):
        for method, (a, b, c) in methods.items():
            train_y = yrand if method == "domain_randomized" else ytr
            curve = fit_curve(a, train_y, b, yid, c, yood, seed)
            for r in curve:
                run_rows.append({"seed": seed, "method": method, **r})

    runs = pd.DataFrame(run_rows)
    final = (runs[runs.epoch == runs.epoch.max()]
             .groupby("method")
             .agg(id_accuracy_mean=("id_accuracy", "mean"), id_accuracy_std=("id_accuracy", "std"),
                  ood_accuracy_mean=("ood_accuracy", "mean"), ood_accuracy_std=("ood_accuracy", "std"))
             .reset_index())
    thresholds = []
    for (seed, method), g in runs.groupby(["seed", "method"]):
        rr = g.sort_values("epoch").to_dict("records")
        thresholds.append({
            "seed": seed, "method": method,
            "epoch_to_80": first_epoch(rr, 0.80),
            "epoch_to_90": first_epoch(rr, 0.90),
        })
    threshold_df = pd.DataFrame(thresholds)
    threshold_summary = (threshold_df.groupby("method")
                         .agg(success_80=("epoch_to_80", lambda x: float(x.notna().mean())),
                              median_epoch_to_80=("epoch_to_80", "median"),
                              success_90=("epoch_to_90", lambda x: float(x.notna().mean())),
                              median_epoch_to_90=("epoch_to_90", "median"))
                         .reset_index())

    runs.to_csv(OUT / "runs.csv", index=False)
    final.to_csv(OUT / "final_summary.csv", index=False)
    threshold_summary.to_csv(OUT / "speed_summary.csv", index=False)
    np.save(OUT / "four_pair_mask.npy", mask4)
    metadata = {
        "runtime_seconds": time.time() - t0,
        "train_examples": len(ytr), "id_examples": len(yid), "ood_examples": len(yood),
        "input_shape": list(Xtr.shape[1:]),
        "four_pair_mask_fraction": float(mask4.mean()),
        "sixtyfour_pair_mask_fraction": float(mask64.mean()),
        "stage_colors": [[0.10, 0.22, 0.85], [0.85, 0.18, 0.10]],
        "train_color_alignment": 0.95, "test_color_reversal": True,
        "note": "Official CausalWorld pixel simulator; supervised goal-side pilot, not full RL leaderboard protocol."
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("FINAL")
    print(final.to_string(index=False))
    print("SPEED")
    print(threshold_summary.to_string(index=False))
    print("META", json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
