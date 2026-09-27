import time
import random

import torch
from sb3_contrib import MaskablePPO

from snake_game_custom_wrapper_cnn import SnakeEnv


GLOBAL_SEED = 114514

if torch.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_snake_final"
else:
    MODEL_PATH = r"trained_models_cnn/ppo_snake_final"

NUM_EPISODE = 10

# 批量测试建议 False，想看画面再改 True
RENDER = True

FRAME_DELAY = 0.05
ROUND_DELAY = 3

ACTION_NAMES = ["UP", "LEFT", "RIGHT", "DOWN"]


def build_test_seed_list(num_episode, base_seed=GLOBAL_SEED):
    rng = random.Random(base_seed)
    return [rng.randint(0, int(1e9)) for _ in range(num_episode)]


def create_env(seed):
    if RENDER:
        return SnakeEnv(
            seed=seed,
            limit_step=False,
            silent_mode=False,
        )

    return SnakeEnv(
        seed=seed,
        limit_step=False,
        silent_mode=True,
    )


def get_env_action_mask(env):
    """
    测试阶段获取 action mask。

    优先使用环境内部的 action_masks()：
        - 返回形状通常是 (4,)
        - 更符合 MaskablePPO 的单环境预测用法

    如果环境没有 action_masks()，再回退到 get_action_mask()。
    """
    if hasattr(env, "action_masks"):
        return env.action_masks()

    return env.get_action_mask().reshape(-1)


def main():
    seed_list = build_test_seed_list(NUM_EPISODE, base_seed=GLOBAL_SEED)

    print(f"Using deterministic test seed sequence derived from GLOBAL_SEED={GLOBAL_SEED}")
    print(f"Episode seeds: {seed_list}")

    model = MaskablePPO.load(MODEL_PATH)

    total_reward = 0.0
    total_score = 0
    min_score = int(1e9)
    max_score = 0

    for episode, episode_seed in enumerate(seed_list, start=1):
        env = create_env(episode_seed)

        obs = env.reset()
        episode_reward = 0.0
        done = False
        num_step = 0
        info = None
        sum_step_reward = 0.0

        print(f"=================== Episode {episode} | seed={episode_seed} ==================")

        while not done:
            action_mask = get_env_action_mask(env)

            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_mask,
            )

            action = int(action)

            num_step += 1
            obs, reward, done, info = env.step(action)

            if done:
                if info["snake_size"] == env.game.grid_size:
                    print(f"You are BREATHTAKING! Victory reward: {reward:.4f}.")
                else:
                    last_action = ACTION_NAMES[action]
                    print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")

            elif info["food_obtained"]:
                print(
                    f"Food obtained at step {num_step:04d}. "
                    f"Food Reward: {reward:.4f}. "
                    f"Accumulated Move Reward: {sum_step_reward:.4f}. "
                    f"Score: {env.game.score}"
                )
                sum_step_reward = 0.0

            else:
                sum_step_reward += reward

            episode_reward += reward

            if RENDER:
                env.render()
                time.sleep(FRAME_DELAY)

        episode_score = env.game.score

        min_score = min(min_score, episode_score)
        max_score = max(max_score, episode_score)

        snake_size = info["snake_size"] + 1

        print(
            f"Episode {episode}: Reward Sum: {episode_reward:.4f}, "
            f"Score: {episode_score}, "
            f"Total Steps: {num_step}, "
            f"Snake Size: {snake_size}"
        )

        total_reward += episode_reward
        total_score += episode_score

        env.close()

        if RENDER:
            time.sleep(ROUND_DELAY)

    print("=================== Summary ==================")
    print(f"Average Score: {total_score / NUM_EPISODE}")
    print(f"Min Score: {min_score}")
    print(f"Max Score: {max_score}")
    print(f"Average Reward: {total_reward / NUM_EPISODE:.4f}")


if __name__ == "__main__":
    main()