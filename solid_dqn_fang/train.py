import os
import sys
import gym
import random
import numpy as np
from xmlrpc.client import ServerProxy
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet
from memory import Memory
from tensorboardX import SummaryWriter

from config import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr
from env_unity import Env

log_dir = 'D:\work\solid_dqn_fang\solid_dqn_fang\logs\car_model.pth' #输出目录改为自己的目录

def get_action(state, target_net, epsilon, env):
    # print("start action")
    if np.random.rand() <= epsilon:
        index = np.random.randint(0, env.action_space_length)
        return index
    else:
        return target_net.get_action(state)

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


def main():
    env = Env('dqn_car')
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = len(env.receive()[0][0])+2
    num_actions = env.action_space_length
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = QNet(num_inputs, num_actions)
    target_net = QNet(num_inputs, num_actions)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    running_score = 0
    epsilon = 1
    steps = 0
    loss = 0

    for e in range(3000):
        done = False

        score = 0
        state = env.reset()
        #print('join state')
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)
        # print(state)
        #print('e in range')
        while not done:
            steps += 1
            #print('join while not')

            #print('state', state)
            #print('target_net', target_net)
            #print('env', env)
            action = get_action(state, target_net, epsilon, env)



            #print('end action!!!!!!!!!')
            real_action = env.action_space[action]

            #print('dongmo',real_action)
            next_state, reward, done, flag = env.step(real_action)
            #print('step')
            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)

            mask = 0 if done else 1
            action_one_hot = np.zeros(env.action_space_length)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

            print('steps', steps)
            print('reward', reward)
            print('score', score)
            print('done', done)

            # if done:
            #     while True:
            #         server = ServerProxy("http://192.168.1.135:6677")  # 初始化服务器
            #         server.t_stop(1)
            #         input("重置小车位置，已成功请按回车")
            #         break


            if flag == '1':
                print("成功标志")
                torch.save(online_net.state_dict(),log_dir)
                break


            if steps > initial_exploration:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(batch_size)
                loss = QNet.train_model(online_net, target_net, optimizer, batch)



                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, score, epsilon))
            writer.add_scalar('log/score', float(score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if flag == '1':
            break
        # if running_score > goal_score:
        #     break




if __name__=="__main__":
    main()
