import time
import random
from snake_game_custom_wrapper_cnn import SnakeEnv

FRAME_DELAY = 0.01  # 设置帧延迟
ROUND_DELAY = 5
BOARD_SIZE = 12

class ModifiedSnakeEnv(SnakeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.advanced_path = self._create_advanced_path()
        self.advanced_path_index = 0

    def _create_advanced_path(self):
        # 创建12x12方格的蛇形路径
        path = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if r % 2 == 0:
                    path.append((r, c))
                else:
                    path.append((r, self.board_size - c - 1))
        return path

    def _get_advanced_strategy_action(self):
        # 根据汉密尔顿环路径决定行动
        current_head = self.game.snake[0]
        next_position = self.advanced_path[self.advanced_path_index]
        self.advanced_path_index = (self.advanced_path_index + 1) % len(self.advanced_path)
        row_diff = next_position[0] - current_head[0]
        col_diff = next_position[1] - current_head[1]
        if row_diff == 0 and col_diff == 1:
            return 2  # 向右
        elif row_diff == 0 and col_diff == -1:
            return 1  # 向左
        elif row_diff == 1 and col_diff == 0:
            return 3  # 向下
        elif row_diff == -1 and col_diff == 0:
            return 0  # 向上
        return -1  # 默认行动

def main():
    seed = random.randint(0, 1e9)
    print(f"Using seed = {seed} for testing.")

    env = ModifiedSnakeEnv(silent_mode=False, seed=seed, board_size=BOARD_SIZE)

    num_step = 0
    done = False

    while not done:
        if env.score >= 1200:
            action = env._get_advanced_strategy_action()
        else:
            action = env.action_space.sample()  # 随机行动

        _, _, done, _ = env.step(action)
        num_step += 1
        env.render()
        time.sleep(FRAME_DELAY)

        if done:
            print(f"Game Finished: Score = {env.game.score}, Total steps = {num_step}")
            time.sleep(ROUND_DELAY)

if __name__ == "__main__":
    main()