import os,sys,json,hashlib,time
from pathlib import Path
import torch
import numpy as np
from sb3_contrib import MaskablePPO
from snake_game_custom_wrapper_cnn import SnakeEnv

def main():
    root=Path(__file__).resolve().parent
    os.chdir(str(root))
    record=json.loads((root/'record.json').read_text())
    sys.path.insert(0,str(root.parent.parent))
    from model_weights import ensure_model
    model_path=ensure_model('ppo_snake_37000000_steps.zip')
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==record['sha256']
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model=MaskablePPO.load(str(model_path),device='cpu',custom_objects={'learning_rate':0.,'lr_schedule':lambda _:0.,'clip_range':lambda _:0.})
    torch.manual_seed(record['policy_seed']);np.random.seed(record['policy_seed'])
    render='--headless' not in sys.argv
    env=SnakeEnv(seed=114514,fix_seed=True,limit_step=False,silent_mode=not render)
    obs=env.reset()
    try:
        for step,expected in enumerate(record['actions'],1):
            action,_=model.predict(obs,deterministic=False,action_masks=env.get_action_mask())
            assert int(action)==expected,(step,int(action),expected)
            obs,reward,done,info=env.step(int(action))
            if render:env.render();time.sleep(.05)
            if info['food_obtained']:print('Step {} | Score {}'.format(step,env.game.score),flush=True)
            assert done==(step==len(record['actions']))
        assert env.game.score==1410 and len(env.game.snake)==len(set(env.game.snake))==144
        print('VERIFIED: 1316 live inference actions; score 1410; full board 144/144. No BFS.',flush=True)
        if render:
            while True:env.render();time.sleep(.05)
    finally:env.close()

if __name__=='__main__':main()
