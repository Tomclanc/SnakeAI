"""直接运行：CNN 固定种子 114514，1316 步满盘；窗口 + 终端输出。"""
import os
from pathlib import Path
import runpy
import subprocess
import sys

def main():
    root=Path(__file__).resolve().parent
    runtime=root/'.snake-runtime/python.exe'
    if runtime.is_file() and Path(sys.executable).resolve()!=runtime.resolve():
        env=os.environ.copy()
        env.pop('PYTHONHOME',None)
        env.pop('PYTHONPATH',None)
        env['PYTHONUTF8']='1'
        return subprocess.call([str(runtime),'-u',str(Path(__file__).resolve())]+sys.argv[1:],env=env)
    assets=root/'replays/fixed_114514'
    if not (assets/'record.json').is_file():
        raise SystemExit('缺少 replays/fixed_114514/model.zip，请保留配套资源目录。')
    sys.path.insert(0,str(assets))
    print('固定种子 114514 | 3700 万步权重 | CNN + 原版防撞 | 无 BFS',flush=True)
    print('现场推理并核对 1316 步通关；结束后关闭游戏窗口退出。',flush=True)
    runpy.run_path(str(assets/'reproduce.py'),run_name='__main__')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
