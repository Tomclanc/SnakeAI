import os
import sys
import random
import multiprocessing as mp

import numpy as np
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO

from snake_game_custom_wrapper_cnn import SnakeEnv


GLOBAL_SEED = 114514

if torch.backends.mps.is_available():
    NUM_ENV = 32 * 2
else:
    NUM_ENV = 32

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 200_000_000


def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    final_value = float(final_value)

    assert initial_value > 0.0

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def build_deterministic_seed_sequence(global_seed, num_env):
    rng = random.Random(global_seed)

    seed_set = set()

    while len(seed_set) < num_env:
        seed_set.add(rng.randint(0, int(1e9)))

    return list(seed_set)


def make_env(seed=0):
    def _init():
        env = SnakeEnv(seed=seed)
        env = Monitor(env)
        env.seed(seed)
        return env

    return _init


def main():
    set_random_seed(GLOBAL_SEED)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    env_seeds = build_deterministic_seed_sequence(
        global_seed=GLOBAL_SEED,
        num_env=NUM_ENV,
    )

    print(f"Using deterministic seed sequence derived from GLOBAL_SEED={GLOBAL_SEED}")
    print(f"First 10 env seeds: {env_seeds[:10]}")

    env = SubprocVecEnv(
        [make_env(seed=s) for s in env_seeds],
        start_method="spawn",
    )

    if torch.backends.mps.is_available():
        device = "mps"

        lr_schedule = linear_schedule(5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)

        batch_size = 512 * 8
        save_dir = "trained_models_cnn_mps"

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)

        batch_size = 512
        save_dir = "trained_models_cnn"

    model = MaskablePPO(
        "CnnPolicy",
        env,
        device=device,
        verbose=1,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=LOG_DIR,
        seed=GLOBAL_SEED,
    )

    os.makedirs(save_dir, exist_ok=True)

    checkpoint_interval = 15625

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=save_dir,
        name_prefix="ppo_snake",
    )

    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")

    try:
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            sys.stdout = log_file

            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=[checkpoint_callback],
            )

    finally:
        sys.stdout = original_stdout
        env.close()

    final_model_path = os.path.join(save_dir, "ppo_snake_final.zip")
    model.save(final_model_path)

    print("Training finished.")
    print(f"Model saved to: {final_model_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()