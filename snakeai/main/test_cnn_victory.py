"""直接运行：原游戏窗口 + 控制台打印，现场复现 2539 步满盘。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parent
# 使用项目内的完整解释器，不依赖旧 uv 路径或虚拟环境的 pyvenv.cfg。
RUNTIME = PROJECT_DIR / '.snake-runtime' / 'python.exe'

EPISODE_SEED = 210750013
MODEL_PATH = PROJECT_DIR / 'trained_models_cnn' / 'ppo_snake_final.zip'
RENDER = True
FRAME_DELAY = 0.05
ACTION_NAMES = ['UP', 'LEFT', 'RIGHT', 'DOWN']

# 仅核对现场推理动作；不会拿记录动作代替模型输出。
EXPECTED_ACTIONS = '3333100111100000023320202202022201113111313333222222022331111010100011111333333333222222000232201011100100133313223323200000000111001333222333323231101001333232222000020002333332000000111111113131001333332322002332322200011101022322233332200000001110011111322231131333332311013322222022220000000001111111131333220220222311311311113333322220220020202200001111111113222222311133331013133332222222222000000000001111111111132222333233332000002233133331111132222220023323200000000010011111111333332322000023231333113113323200202200020233133311322200020000110101111113133333333320000233332000233320000110222333322220000000001333111113100000133333113333332000233320023320001102020233323223200000000001113223333333100000011013101310133223113333333200000023333332222222200000000000133333331323110002000001133333333111020000000131013313223113332313222222222200000000001333333333100001020001331333233101310020000101310131013323133333332222010010023233233222222000000000001333333333310000000000133331000011111132232232332233331100010011111333202331132220002333232223222000000000001333333333310000000000111111111322222222333100133310001111133333200002333331133322220000233320000233333222200013310002200000001333331000001333310001333313110200000111132233333323331001333222000023333200000233333222201110222011022000000013333331000000111133223131310000011113223113320233113333200023332000233332000020233133323200233200011022200000001333333100000013101332313100013100111323133202223113113333332201000023332000022222311133132320023320002333200020000000133333310000001333100011132233310011001331001333231333333322010000023333200002222023311113333132222201022332201020111022200000001333331000001331001333311000013333100001133333333333220100000233332000022220233113113331322000233320023320023320001102200000001333333100000011313133100001333310000113333333333320000002333320000232020233231111333113222000233320002333200233200010200000001333331000001113132223111310000133331000013333133333332000000233320002220223131113331132320200023332000202333323200010020000001333331000001310133223111310000133331000013333133333332000000233320002220223131113331133202320000233332000023333200233200011022000000013333331000000133310001331001331001113332002332231111322313220022002322331013113311133320023320020023333200002333320023320001102200000001333333100000013331000133100133100111332023322311101332231132220022002322311311331113313220023322010200233332000002333332002332000110220000000133333310000001333100013310013310011133202332231110133223113222002200232231131133111132'


def get_tail_guard_mask(env):
    import numpy as np
    instant = np.array([env._check_action_validity(a) for a in range(4)], dtype=bool)
    if env.game.score < 400:
        return instant
    safe = np.zeros(4, dtype=bool)
    for action in range(4):
        if not instant[action]:
            continue
        snake, _ = env._simulate_action(action)
        safe[action] = len(snake) == env.grid_size or env._bfs_reachable_on_state(
            snake[0], snake[-1], set(snake[1:-1]))
    return safe if safe.any() else instant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='只核验并打印，不显示窗口')
    parser.add_argument('--frame-delay', type=float, default=FRAME_DELAY, help='每步等待秒数，默认 0.05')
    parser.add_argument('--exit-on-finish', action='store_true', help='结束后关闭窗口；默认保留满盘画面')
    args = parser.parse_args()
    if args.frame_delay < 0:
        parser.error('--frame-delay 不能为负数')

    import hashlib
    import numpy as np
    import torch
    import pygame
    from sb3_contrib import MaskablePPO
    from snake_game_custom_wrapper_cnn import SnakeEnv

    os.chdir(str(PROJECT_DIR))  # 原游戏的 sound/ 资源使用相对路径。
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(271828)
    np.random.seed(271828)
    model_sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if model_sha != '81f64d949bf5c1a11f73faa8eb599d0137fedfda248e8c51ee9eebf90a05e2a5':
        raise RuntimeError('当前模型不是已核验的权重，无法保证复现这局。')

    print('=================== LIVE CNN VICTORY ===================', flush=True)
    print('Python: {}'.format(sys.executable), flush=True)
    print('Episode seed: {} | CPU | deterministic=True'.format(EPISODE_SEED), flush=True)
    print('Expected: 2539 steps, score 1410, snake size 144.', flush=True)
    print('Live model inference + tail guard; recorded actions are only checked.', flush=True)
    model = MaskablePPO.load(str(MODEL_PATH), device='cpu', custom_objects={
        'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0, 'clip_range': lambda _: 0.0})
    render = RENDER and not args.headless
    env = SnakeEnv(seed=EPISODE_SEED, limit_step=False, silent_mode=not render)
    # 与 test_cnn.py 相同：构造环境后再 reset 一次，不要额外 reset。
    obs = env.reset()
    total_reward = 0.0
    move_reward = 0.0
    step = 0
    done = False
    try:
        if render:
            env.render()
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=get_tail_guard_mask(env))
            action = int(action)
            if step >= len(EXPECTED_ACTIONS) or action != int(EXPECTED_ACTIONS[step]):
                raise RuntimeError('第 {} 步现场推理与参考路线不同，已停止。'.format(step + 1))
            obs, reward, done, info = env.step(action)
            step += 1
            total_reward += reward
            if info['food_obtained']:
                print('Food obtained at step {:04d}. Food Reward: {:.4f}. '
                      'Accumulated Move Reward: {:.4f}. Score: {}'.format(
                          step, reward, move_reward, env.game.score), flush=True)
                move_reward = 0.0
            else:
                move_reward += reward
            if render:
                env.render()
                time.sleep(args.frame_delay)

        assert info['end_reason'] == 'victory'
        assert step == 2539 and env.game.score == 1410
        assert len(env.game.snake) == len(set(env.game.snake)) == 144
        print('You are BREATHTAKING! Victory reward: {:.4f}.'.format(reward), flush=True)
        print('=================== Summary ===================', flush=True)
        print('Reward Sum: {:.4f}, Score: {}, Total Steps: {}, Snake Size: {}'.format(
            total_reward, env.game.score, step, len(env.game.snake)), flush=True)
        print('VERIFIED: all 2539 live actions match. Full board: 144/144.', flush=True)
        if render and not args.exit_on_finish:
            print('Close the game window to exit.', flush=True)
            while True:
                env.render()
                time.sleep(0.05)
    finally:
        pygame.quit()
        env.close()


if __name__ == '__main__':
    if RUNTIME.is_file() and Path(sys.executable).resolve() != RUNTIME.resolve():
        if not RUNTIME.is_file():
            raise SystemExit('缺少项目内的 .snake-runtime/python.exe，请保留该运行环境目录。')
        child_env = os.environ.copy()
        child_env.pop('PYTHONHOME', None)
        child_env.pop('PYTHONPATH', None)
        child_env['PYTHONUTF8'] = '1'
        raise SystemExit(subprocess.call([str(RUNTIME), str(Path(__file__).resolve())] + sys.argv[1:], env=child_env))
    main()
