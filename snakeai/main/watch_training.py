"""Compact live view of the existing training log; does not start training."""
import argparse
from pathlib import Path
import time

def summary(v):
    steps=int(float(v['total_timesteps']))
    fps=float(v.get('fps',0))
    elapsed=int(float(v.get('time_elapsed',0)))
    remaining=max(0,100000000-steps)/fps if fps else 0
    return ('进度 {:6.2f}% | {:,}/1亿步 | 平均奖励 {} | 平均局长 {}步 | '
            '{:.0f}步/秒 | 已用 {:.1f}分钟 | 预计剩余 {:.1f}小时').format(
                steps/1000000,steps,v.get('ep_rew_mean','?'),v.get('ep_len_mean','?'),
                fps,elapsed/60,remaining/3600)

class CompactConsole:
    def __init__(self,stream):
        self.stream=stream
        self.buffer=''
        self.values={}
    def write(self,text):
        self.buffer+=text
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            stripped=line.strip()
            if stripped.startswith('|'):
                parts=[x.strip() for x in stripped.split('|')]
                if len(parts)>=4 and parts[1] and parts[2]:self.values[parts[1]]=parts[2]
            elif stripped and set(stripped)=={'-'}:
                if 'total_timesteps' in self.values:
                    self.stream.write(summary(self.values)+'\n')
                    self.values={}
            else:
                self.stream.write(line+'\n')
        self.flush()
        return len(text)
    def flush(self):self.stream.flush()

def main():
    import sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    logs=list((Path(__file__).resolve().parent/'runs').glob('*/trained_models_02/training_log.txt'))
    if not logs:raise SystemExit('未找到训练日志。')
    path=max(logs,key=lambda p:p.stat().st_mtime)
    print('查看日志：'+str(path),flush=True)
    print('平均奖励不是游戏分数；预计剩余时间会随训练速度变化。',flush=True)
    output=CompactConsole(sys.stdout)
    with path.open('r',encoding='utf-8',errors='replace') as f:
        while True:
            output.write(f.read())
            if args.once:return
            time.sleep(2)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:pass
