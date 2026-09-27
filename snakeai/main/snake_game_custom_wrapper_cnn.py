import math
from collections import deque

import gym
import numpy as np

from snake_game import SnakeGame


class SnakeEnv(gym.Env):
    """
    奖励设计（12x12，满分 1410）：
    - 400 分之前：保持原版风格基本不变，只 mask 即时死亡动作
    - 400 分之后：加入轻量未来安全检测，减少中期提前自锁
    - 600 分之后：降低“急着贴近果子”的偏好，增加存活与安全性权重
    - 800 分之后：明显加强 future safety mask，减少 800~1000 分把空间结构走坏
    - 900 分之后：进一步限制空间切割，尽量为 1000+ 后期保留完整空间
    - 1000 分之后：强化安全拓扑，避免继续抄近路
    - 1200 分之后：极后期更保守，为后续汉密尔顿收尾预留空间

    本版关键改动：
    1. 不再折腾 300~399 分，因为上一版 TensorBoard 基本证明这个阶段影响很小。
    2. 保留 PPO_2 有效的 400 分启动 future safety mask。
    3. 把真正有效干预提前到 800~999 分。
    4. 800~899 分比 PPO_2 更重视空间比例和区域连通。
    5. 900~999 分进一步减少空白区域被切碎。
    6. 1000+ 继续保守，但不只依赖 1000+，避免“改得太晚导致训练曲线不变”。
    7. 极端情况下如果所有未来安全动作都被禁掉，会回退到即时合法动作，避免 MaskablePPO 遇到全 False mask。
    """

    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True):
        super().__init__()

        self.seed_value = seed
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()

        self.silent_mode = silent_mode

        # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        self.action_space = gym.spaces.Discrete(4)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 3),
            dtype=np.uint8,
        )

        self.board_size = board_size
        self.grid_size = board_size ** 2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False
        self.use_step_limit = limit_step
        self.default_step_limit = self.grid_size * 4

        self.reward_step_counter = 0

    def seed(self, seed=None):
        """
        兼容 Stable-Baselines3 / Monitor 的 seed 调用。
        """
        if seed is None:
            seed = self.seed_value

        self.seed_value = int(seed)
        self.game = SnakeGame(
            seed=self.seed_value,
            board_size=self.board_size,
            silent_mode=self.silent_mode,
        )
        self.game.reset()

        return [self.seed_value]

    def reset(self):
        self.game.reset()
        self.done = False
        self.reward_step_counter = 0
        obs = self._generate_observation()
        return obs

    def step(self, action):
        prev_direction = self.game.direction
        prev_snake = list(self.game.snake)
        prev_head = prev_snake[0]
        prev_food = self.game.food

        self.done, info = self.game.step(action)

        obs = self._generate_observation()
        self.reward_step_counter += 1

        current_score = self.game.score
        stage = self._get_stage_by_score(current_score)

        info["end_reason"] = None

        # 满盘胜利
        if info["snake_size"] == self.grid_size:
            reward = 25.0
            self.done = True
            info["end_reason"] = "victory"

            if not self.silent_mode:
                self.game.sound_victory.play()

            return obs, reward, self.done, info

        # 动态步数上限：限制“连续多少步没吃到果子”
        current_step_limit = self._get_current_step_limit(current_score)

        if self.use_step_limit and self.reward_step_counter > current_step_limit:
            self.reward_step_counter = 0
            self.done = True
            info["end_reason"] = "timeout"

        # 游戏结束但不是 victory / timeout，则判断 wall / self
        if self.done and info["end_reason"] is None:
            info["end_reason"] = self._classify_collision_reason(
                action=action,
                prev_direction=prev_direction,
                prev_snake=prev_snake,
                prev_head=prev_head,
                prev_food=prev_food,
            )

        # 死亡 / 结束惩罚
        if self.done:
            reward = self._calculate_death_reward(info, stage)
            return obs, reward, self.done, info

        # 吃到食物
        if info["food_obtained"]:
            reward = self._calculate_food_reward(info, stage)
            self.reward_step_counter = 0
            return obs, reward, self.done, info

        # 普通移动
        reward = self._calculate_move_reward(info, stage)
        return obs, reward, self.done, info

    def render(self):
        self.game.render()

    def get_action_mask(self):
        """
        返回形状为 (1, 4) 的动作 mask。

        400 分前：
            只判断下一步是否立即撞墙 / 撞身体 / 反向。

        400 分后：
            在即时合法基础上，模拟动作后的局面，
            检查头到尾可达、头部可达空间、空间比例等。

        极端情况：
            如果严格安全检测把所有动作都禁掉，
            回退到即时合法动作，避免所有 mask 都是 False。
        """
        instant_valid_mask = [
            self._check_action_validity(action)
            for action in range(self.action_space.n)
        ]

        # 400 分前不强行限制未来安全，保留 PPO 的观赏性和探索空间
        if self.game.score < 400:
            return np.array([instant_valid_mask], dtype=bool)

        future_safe_mask = []

        for action in range(self.action_space.n):
            if not instant_valid_mask[action]:
                future_safe_mask.append(False)
                continue

            future_safe = self._check_action_future_safety(action)
            future_safe_mask.append(future_safe)

        # 如果存在未来安全动作，使用更严格的 future_safe_mask
        if any(future_safe_mask):
            return np.array([future_safe_mask], dtype=bool)

        # 如果严格安全检测导致全部 False，则回退到即时合法动作
        return np.array([instant_valid_mask], dtype=bool)

    def action_masks(self):
        """
        MaskablePPO 在 SubprocVecEnv 下推荐环境内部实现 action_masks()。
        返回形状为 (4,) 的布尔数组。
        """
        return self.get_action_mask().reshape(-1)

    def _get_stage_by_score(self, score):
        """
        分段：
        - stage 0: 600 分之前
        - stage 1: 600 ~ 990 分
        - stage 2: 1000 分及以上
        """
        if score >= 1000:
            return 2

        if score >= 600:
            return 1

        return 0

    def _get_current_step_limit(self, score):
        """
        连续未吃到果子的动态步数阈值：
        - 600 分前：576
        - 600 分后：600
        - 1000 分后：800
        """
        if not self.use_step_limit:
            return int(1e9)

        if score >= 1000:
            return 800

        if score >= 600:
            return 600

        return self.default_step_limit

    def _calculate_death_reward(self, info, stage):
        """
        基础死亡惩罚 + 分阶段放大 + 按结束原因微调
        """
        base_penalty = -math.pow(
            self.max_growth,
            (self.grid_size - info["snake_size"]) / self.max_growth,
        ) * 0.1

        if stage == 0:
            stage_multiplier = 1.0
        elif stage == 1:
            stage_multiplier = 1.6
        else:
            stage_multiplier = 2.4

        end_reason = info.get("end_reason", "wall")

        if end_reason == "self":
            reason_multiplier = 1.35
        elif end_reason == "wall":
            reason_multiplier = 1.0
        elif end_reason == "timeout":
            reason_multiplier = 0.90
        else:
            reason_multiplier = 1.0

        return base_penalty * stage_multiplier * reason_multiplier

    def _calculate_food_reward(self, info, stage):
        """
        吃食物奖励
        """
        base_reward = info["snake_size"] / self.grid_size

        if stage == 0:
            multiplier = 1.0
        elif stage == 1:
            multiplier = 1.0
        else:
            multiplier = 1.05

        return base_reward * multiplier

    def _calculate_move_reward(self, info, stage):
        """
        普通移动奖励包含四部分：
        1. 靠近/远离食物奖励
        2. 存活奖励
        3. 基础安全性奖励（头到尾、空白区连通）
        4. 生存空间奖励（头部可达空间、下一步动作数、空间相对蛇长）
        """
        current_dist = np.linalg.norm(info["snake_head_pos"] - info["food_pos"])
        prev_dist = np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"])

        base_distance_reward = (1 / info["snake_size"]) * 0.1

        if current_dist < prev_dist:
            move_toward_food_reward = base_distance_reward
        else:
            move_toward_food_reward = -base_distance_reward

        if stage == 0:
            distance_multiplier = 1.0
            survive_reward = 0.0
            safety_reward = 0.0
            living_space_reward = 0.0

        elif stage == 1:
            distance_multiplier = 0.45
            survive_reward = 0.006
            safety_reward = self._calculate_safety_reward(stage=1)
            living_space_reward = self._calculate_living_space_reward(stage=1)

        else:
            distance_multiplier = 0.15
            survive_reward = 0.010
            safety_reward = self._calculate_safety_reward(stage=2)
            living_space_reward = self._calculate_living_space_reward(stage=2)

        reward = move_toward_food_reward * distance_multiplier
        reward += survive_reward
        reward += safety_reward
        reward += living_space_reward

        return reward

    def _calculate_safety_reward(self, stage):
        """
        基础安全性奖励：
        1. 头到尾可达：正奖励
        2. 空白区被切割得更碎：轻微惩罚
        """
        head_to_tail_reachable = self._is_head_to_tail_reachable()
        free_region_count = self._count_free_regions()

        if stage == 1:
            reachable_bonus = 0.015 if head_to_tail_reachable else -0.020
            region_penalty = max(0, free_region_count - 1) * 0.004
        else:
            reachable_bonus = 0.030 if head_to_tail_reachable else -0.045
            region_penalty = max(0, free_region_count - 1) * 0.008

        return reachable_bonus - region_penalty

    def _calculate_living_space_reward(self, stage):
        """
        生存空间检测：
        1. 头部可达空间大小
        2. 下一步可行动作数量
        3. 空间相对蛇长是否足够
        """
        head_region_size = self._head_reachable_area_size()
        next_valid_action_count = self._count_valid_actions_current_state()
        snake_len = len(self.game.snake)

        space_ratio = head_region_size / max(1, snake_len)

        if stage == 1:
            area_bonus = (head_region_size / self.grid_size) * 0.020

            if next_valid_action_count >= 3:
                action_bonus = 0.012
            elif next_valid_action_count == 2:
                action_bonus = 0.004
            elif next_valid_action_count == 1:
                action_bonus = -0.012
            else:
                action_bonus = -0.030

            if space_ratio < 1.1:
                ratio_penalty = -0.018
            elif space_ratio < 1.4:
                ratio_penalty = -0.008
            else:
                ratio_penalty = 0.0

        else:
            area_bonus = (head_region_size / self.grid_size) * 0.040

            if next_valid_action_count >= 3:
                action_bonus = 0.020
            elif next_valid_action_count == 2:
                action_bonus = 0.006
            elif next_valid_action_count == 1:
                action_bonus = -0.020
            else:
                action_bonus = -0.050

            if space_ratio < 1.15:
                ratio_penalty = -0.030
            elif space_ratio < 1.5:
                ratio_penalty = -0.012
            else:
                ratio_penalty = 0.0

        return area_bonus + action_bonus + ratio_penalty

    def _check_action_future_safety(self, action):
        """
        模拟执行某个动作后，判断这个动作是否会把蛇带入未来死局。

        检查项：
        1. 动作本身是否合法
        2. 执行动作后的蛇头是否还能到达蛇尾
        3. 执行动作后的头部可达区域大小
        4. 可达空间 / 蛇长 的比例
        5. 中后期额外检查空白区域是否被切碎

        返回：
            True  = 允许这个动作
            False = mask 掉这个动作
        """
        simulated = self._simulate_action(action)

        if simulated is None:
            return False

        new_snake, ate_food = simulated

        new_head = new_snake[0]
        new_tail = new_snake[-1]
        new_len = len(new_snake)

        remaining_empty = self.grid_size - new_len

        if new_len >= self.grid_size:
            return True

        # 阻塞格：
        # - 头不算 blocked
        # - 尾巴视作未来可释放，所以尾巴不算 blocked
        # - 中间身体算 blocked
        blocked = set(new_snake[1:-1])

        head_to_tail_reachable = self._bfs_reachable_on_state(
            start=new_head,
            target=new_tail,
            blocked=blocked,
        )

        head_area_size = self._bfs_area_size_on_state(
            start=new_head,
            blocked=blocked,
        )

        space_ratio = head_area_size / max(1, new_len)

        score = self.game.score

        # 400 ~ 599：
        # 保留 PPO_2 有效逻辑：轻量安全检测，主要防止中期提前把空间切坏。
        if 400 <= score < 600:
            if head_to_tail_reachable:
                return True

            if space_ratio >= 1.45 and head_area_size >= new_len + 12:
                return True

            return False

        # 600 ~ 749：
        # 保持原 PPO_2 风格，不太早牺牲观赏性和吃果效率。
        if 600 <= score < 750:
            if head_to_tail_reachable:
                return True

            if space_ratio >= 1.35 and head_area_size >= new_len + 8:
                return True

            return False

        # 750 ~ 799：
        # 轻微提前收紧，为 800+ 阶段做空间准备。
        if 750 <= score < 800:
            if head_to_tail_reachable:
                if remaining_empty <= 18:
                    return True

                if space_ratio >= 1.08:
                    return True

                return False

            if space_ratio >= 1.45 and head_area_size >= new_len + 10:
                return True

            return False

        # 800 ~ 899：
        # 本版重点改动之一：
        # 这里开始明显加强限制，避免 800 分后继续抄近路导致 900~1000 分自锁。
        if 800 <= score < 900:
            if not head_to_tail_reachable:
                return False

            if remaining_empty <= 14:
                return True

            if space_ratio < 1.12:
                return False

            free_region_count = self._count_free_regions_on_state(new_snake)

            if free_region_count >= 3 and remaining_empty > 14:
                return False

            return True

        # 900 ~ 999：
        # 本版重点改动之二：
        # 进一步减少空间切割，尽量保证 1000+ 后还能追尾 / 收尾。
        if 900 <= score < 1000:
            if not head_to_tail_reachable:
                return False

            if remaining_empty <= 12:
                return True

            if space_ratio < 1.18:
                return False

            free_region_count = self._count_free_regions_on_state(new_snake)

            if free_region_count >= 2 and remaining_empty > 12:
                return False

            return True

        # 1000 ~ 1099：
        # 后期开始明显保守。
        if 1000 <= score < 1100:
            if not head_to_tail_reachable:
                return False

            if remaining_empty <= 10:
                return True

            if space_ratio < 1.22:
                return False

            free_region_count = self._count_free_regions_on_state(new_snake)

            if free_region_count >= 2 and remaining_empty > 10:
                return False

            return True

        # 1100 ~ 1199：
        # 更接近收尾，继续提高空间完整性要求。
        if 1100 <= score < 1200:
            if not head_to_tail_reachable:
                return False

            if remaining_empty <= 8:
                return True

            if space_ratio < 1.25:
                return False

            free_region_count = self._count_free_regions_on_state(new_snake)

            if free_region_count >= 2 and remaining_empty > 8:
                return False

            return True

        # 1200 分以后：
        # 极后期基本进入收尾，要求更稳。
        # 但最后空格很少时，不再机械要求 space_ratio。
        if score >= 1200:
            if not head_to_tail_reachable:
                return False

            if remaining_empty <= 6:
                return True

            if space_ratio < 1.30:
                return False

            free_region_count = self._count_free_regions_on_state(new_snake)

            if free_region_count >= 2 and remaining_empty > 6:
                return False

            return True

        return True

    def _simulate_action(self, action):
        """
        在不修改真实游戏状态的情况下，模拟执行一个动作。

        返回：
            (new_snake, ate_food)

        如果动作非法，返回 None。
        """
        current_direction = self.game.direction
        snake = list(self.game.snake)
        head_row, head_col = snake[0]

        if action == 0:  # UP
            if current_direction == "DOWN":
                return None
            new_head = (head_row - 1, head_col)

        elif action == 1:  # LEFT
            if current_direction == "RIGHT":
                return None
            new_head = (head_row, head_col - 1)

        elif action == 2:  # RIGHT
            if current_direction == "LEFT":
                return None
            new_head = (head_row, head_col + 1)

        elif action == 3:  # DOWN
            if current_direction == "UP":
                return None
            new_head = (head_row + 1, head_col)

        else:
            return None

        row, col = new_head

        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return None

        ate_food = new_head == self.game.food

        if ate_food:
            # 吃到食物时，尾巴不会移动，所以整个蛇身都算占用。
            if new_head in snake:
                return None

            new_snake = [new_head] + snake

        else:
            # 没吃到食物时，尾巴会释放，所以允许走到当前尾巴格。
            if new_head in snake[:-1]:
                return None

            new_snake = [new_head] + snake[:-1]

        return new_snake, ate_food

    def _is_head_to_tail_reachable(self):
        """
        判断当前局面下，蛇头是否还能通过“空格 + 蛇尾位置”到达蛇尾。
        """
        snake = list(self.game.snake)

        if len(snake) <= 2:
            return True

        head = snake[0]
        tail = snake[-1]

        blocked = set(snake[:-1])
        blocked.discard(head)

        return self._bfs_reachable(
            start=head,
            target=tail,
            blocked=blocked,
        )

    def _head_reachable_area_size(self):
        """
        统计从蛇头出发，在“空格 + 蛇尾可释放位置”条件下，
        最终可到达的区域大小。
        """
        snake = list(self.game.snake)
        head = snake[0]

        blocked = set(snake[:-1])
        blocked.discard(head)

        return self._bfs_area_size(start=head, blocked=blocked)

    def _count_free_regions(self):
        """
        统计当前“空白格 + 蛇尾”形成的连通区域数量。
        """
        snake = set(self.game.snake[:-1])
        visited = set()
        regions = 0

        for row in range(self.board_size):
            for col in range(self.board_size):
                pos = (row, col)

                if pos in snake or pos in visited:
                    continue

                regions += 1

                queue = deque([pos])
                visited.add(pos)

                while queue:
                    current = queue.popleft()

                    for nxt in self._neighbors(current):
                        if nxt in snake or nxt in visited:
                            continue

                        visited.add(nxt)
                        queue.append(nxt)

        return regions

    def _count_valid_actions_current_state(self):
        """
        统计当前状态下，下一步有多少即时合法动作。
        """
        count = 0

        for action in range(self.action_space.n):
            if self._check_action_validity(action):
                count += 1

        return count

    def _bfs_reachable(self, start, target, blocked):
        queue = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()

            if current == target:
                return True

            for nxt in self._neighbors(current):
                if nxt in blocked or nxt in visited:
                    continue

                visited.add(nxt)
                queue.append(nxt)

        return False

    def _bfs_area_size(self, start, blocked):
        queue = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()

            for nxt in self._neighbors(current):
                if nxt in blocked or nxt in visited:
                    continue

                visited.add(nxt)
                queue.append(nxt)

        return len(visited)

    def _bfs_reachable_on_state(self, start, target, blocked):
        """
        在模拟蛇身状态上判断 start 是否能到达 target。
        """
        queue = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()

            if current == target:
                return True

            for nxt in self._neighbors(current):
                if nxt in blocked or nxt in visited:
                    continue

                visited.add(nxt)
                queue.append(nxt)

        return False

    def _bfs_area_size_on_state(self, start, blocked):
        """
        在模拟蛇身状态上统计 start 可达区域大小。
        """
        queue = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()

            for nxt in self._neighbors(current):
                if nxt in blocked or nxt in visited:
                    continue

                visited.add(nxt)
                queue.append(nxt)

        return len(visited)

    def _count_free_regions_on_state(self, snake):
        """
        在模拟蛇身状态上统计空白区域数量。

        尾巴视作未来可释放位置，所以不把尾巴作为 blocked。
        头部所在位置视为当前占用，空白区域统计只关心剩余区域是否被切碎。
        """
        if not snake:
            return 1

        blocked = set(snake[:-1])
        visited = set()
        regions = 0

        for row in range(self.board_size):
            for col in range(self.board_size):
                pos = (row, col)

                if pos in blocked or pos in visited:
                    continue

                regions += 1

                queue = deque([pos])
                visited.add(pos)

                while queue:
                    current = queue.popleft()

                    for nxt in self._neighbors(current):
                        if nxt in blocked or nxt in visited:
                            continue

                        visited.add(nxt)
                        queue.append(nxt)

        return regions

    def _neighbors(self, pos):
        row, col = pos

        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        valid = []

        for next_row, next_col in candidates:
            if 0 <= next_row < self.board_size and 0 <= next_col < self.board_size:
                valid.append((next_row, next_col))

        return valid

    def _classify_collision_reason(self, action, prev_direction, prev_snake, prev_head, prev_food):
        """
        在 episode 结束且不是 victory/timeout 时，尽量判断是撞墙还是咬到自己。
        """
        attempted_head = self._get_attempted_head_position(
            action=action,
            prev_direction=prev_direction,
            prev_head=prev_head,
        )

        row, col = attempted_head

        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return "wall"

        if attempted_head == prev_food:
            occupied = set(prev_snake)
        else:
            occupied = set(prev_snake[:-1])

        if attempted_head in occupied:
            return "self"

        return "wall"

    def _get_attempted_head_position(self, action, prev_direction, prev_head):
        row, col = prev_head

        if action == 0:  # UP
            if prev_direction == "DOWN":
                row += 1
            else:
                row -= 1

        elif action == 1:  # LEFT
            if prev_direction == "RIGHT":
                col += 1
            else:
                col -= 1

        elif action == 2:  # RIGHT
            if prev_direction == "LEFT":
                col -= 1
            else:
                col += 1

        elif action == 3:  # DOWN
            if prev_direction == "UP":
                row -= 1
            else:
                row += 1

        return row, col

    def _check_action_validity(self, action):
        """
        即时合法性判断：
        - 不能反向
        - 不能出界
        - 不能撞身体
        - 如果没吃食物，允许走到当前尾巴位置
        - 如果吃食物，尾巴不会移动，所以不能走到任何蛇身位置
        """
        current_direction = self.game.direction
        snake_list = list(self.game.snake)
        row, col = snake_list[0]

        if action == 0:  # UP
            if current_direction == "DOWN":
                return False
            row -= 1

        elif action == 1:  # LEFT
            if current_direction == "RIGHT":
                return False
            col -= 1

        elif action == 2:  # RIGHT
            if current_direction == "LEFT":
                return False
            col += 1

        elif action == 3:  # DOWN
            if current_direction == "UP":
                return False
            row += 1

        else:
            return False

        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False

        new_head = (row, col)

        if new_head == self.game.food:
            if new_head in snake_list:
                return False
        else:
            if new_head in snake_list[:-1]:
                return False

        return True

    # EMPTY: BLACK
    # Snake BODY: GRAY/GREEN GRADIENT
    # Snake HEAD: GREEN
    # Snake TAIL: RED
    # FOOD: BLUE
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        obs[tuple(np.transpose(self.game.snake))] = np.linspace(
            200,
            50,
            len(self.game.snake),
            dtype=np.uint8,
        )

        obs = np.stack((obs, obs, obs), axis=-1)

        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]
        obs[self.game.food] = [0, 0, 255]

        obs = np.repeat(np.repeat(obs, 7, axis=0), 7, axis=1)

        return obs