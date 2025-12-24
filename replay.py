import random, datetime
from pathlib import Path

import gym,os,time
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
import argparse
from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame
envpath = '/home/xumin/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

# python replay.py --agent Nature_DQN --model xxx.chkpt
# python replay.py --agent Nature_DQN_RNN --model xxx.chkpt
# python replay.py --agent Nature_DQN_Transformer --model xxx.chkpt
# python replay.py --agent Double_DQN --model xxx.chkpt
# python replay.py --agent Dueling_DQN --model xxx.chkpt


parser = argparse.ArgumentParser()
parser.add_argument('--agent', metavar='ARCH', default='Nature_DQN')
parser.add_argument('--model', metavar='M', default=' ')
args = parser.parse_args()

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)


trained_model = args.model
checkpoint = Path(trained_model)
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, agent_type=args.agent, checkpoint=checkpoint)

mario.exploration_rate = mario.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 10

start_time = time.time()
for e in range(episodes):

    state = env.reset()

    while True:

        env.render()

        action = mario.act(state)

        next_state, reward, done, info = env.step(action)

        mario.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None)

        state = next_state

        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 1 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )


end_time = time.time()
test_time = end_time - start_time
print('\n')
print('Ave_test_time:',test_time / float(episodes))
env.close()
