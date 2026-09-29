import os
import sys
import random
from pathlib import Path
from datetime import datetime
import torch
from watch_training import CompactConsole

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams: stream.write(value)
        self.flush()
    def flush(self):
        for stream in self.streams: stream.flush()


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_custom_wrapper_cnn import SnakeEnv

NUM_ENV = 32
LOG_DIR = "logs"

FIX_SEED = True
SEED_VALUE = 114514



# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(seed=0):
    def _init():
        if FIX_SEED:
            env = SnakeEnv(seed=SEED_VALUE, fix_seed=True)
        else:
            env = SnakeEnv(seed=seed)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        if not FIX_SEED:
            env.seed(seed)
        return env
    return _init

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate the SnakeAI environment.")
    torch.set_num_threads(1)
    run_dir = Path(__file__).resolve().parent / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    os.chdir(str(run_dir))
    os.makedirs(LOG_DIR, exist_ok=True)
    print('Output:', run_dir, flush=True)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Fresh CNN | fixed food seed 114514 | 32 environments | 100,000,000 steps', flush=True)


    # Generate a list of random seeds for each environment.
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))

    # Create the Snake environment.
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])

    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # Instantiate a PPO agent
    model = MaskablePPO(
        "CnnPolicy",
        env,
        device="cuda",
        verbose=1,
        n_steps=2048,
        batch_size=512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=LOG_DIR
    )

    # continue
    # lr_schedule = linear_schedule(2e-4, 2.5e-6)
    # clip_range_schedule = linear_schedule(0.1205, 0.025)

    # custom_objects = {
    #     "learning_rate": lr_schedule,
    #     "clip_range": clip_range_schedule
    # }

    # model_path = "trained_models/ppo_snake_23000000_steps.zip"
    # model = MaskablePPO.load(model_path, env=env, device="cuda", custom_objects=custom_objects)

    # Set the save directory
    save_dir = "trained_models_02"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_interval = 15625 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_snake")

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        sys.stdout = Tee(CompactConsole(original_stdout), log_file)
        try:
            model.learn(
                total_timesteps=int(100000000),
                callback=[checkpoint_callback]
            )
        finally:
            sys.stdout = original_stdout
            env.close()

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_snake_final.zip"))

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    main()
