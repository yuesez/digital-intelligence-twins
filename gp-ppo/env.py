import numpy as np
import sys
from socket import *
import struct
#import socket
import math
import itertools
from xmlrpc.client import ServerProxy
from gym.utils import seeding
import time

class Env(object):

    def __init__(self, name):
        self.env_name = name
        self.start = 0

        self.ini = np.array([-0.1614985437835034, -1.8565765961139282, -1.15425649427304])  # 起始坐标

        self.end = np.array([-0.7723647425804135, -1.8461237252470015, -1.9526432592840006])  # 终点坐标

        self.waypoint = np.array([[-0.1614985437835034, -1.8565765961139282, -1.15425649427304],  # 中间路线点，2点
                                  [-0.7723647425804135, -1.8461237252470015, -1.9526432592840006]])

        self.lane_bound = np.array([[-0.24735840852735302, -1.8584818326511996, -1.1099386050278366],  # 赛道边界，左2点，右2点
                                    [-0.8319140414849677, -1.8483574187217244, -1.8795862720562337],
                                    [-0.08594792005668857, -1.854484159243324, -1.2125489810466437],
                                    [-0.6864453254765356, -1.844248611330991, -1.995533245747086]])

        #self.end_bound = np.array([[-1.8683, -1.8706, -2.3688],  # 终点路线点， 2点
                                   #[-3.3041, -1.8046, -4.2509]])

        self.race = 1.0053  # 赛道长度

        self.wide = 0.1913  # 赛道宽度

        # get action spapce
        # steer: [-1, -0.5, 0, 0.5, 1], acceleration: [0.0, 1.0]
        self.steer = [-1, -0.5, 0, 0.5, 1]
        self.acc = [5.0, 10.0]
        vectors = [self.steer, self.acc]
        combs = list(itertools.product(*[range(len(v)) for v in vectors]))
        self.action_space = np.array([[vectors[vi][ci] for vi, ci in enumerate(comb)] for comb in combs])
        self.action_space_length = len(self.action_space)

    def rule(self, action):
        steer = action[0]
        acc = action[1]
        if steer == -1:
            command = 'left'
        if steer == -0.5:
            command = 'left_up'
        if steer == 0:
            command = 'up'
        if steer == 0.5:
            command = 'right_up'
        if steer == 1:
            command = 'right'
        if acc == 5:
            velocity = 30
        if acc == 10:
            velocity = 50

        return [command, velocity]
        
    def back_rule(self, action):
        steer = action[0]
        acc = action[1]
        if steer == -1:
            command = 'right'
        if steer == -0.5:
            command = 'right_down'
        if steer == 0:
            command = 'down'
        if steer == 0.5:
            command = 'left_down'
        if steer == 1:
            command = 'left'
        if acc == 5:
            velocity = 30
        if acc == 10:
            velocity = 50

        return [command, velocity]
    #作为客户端接收unity的坐标信息
    def receive(self):
        #while True:
        recvSocket = socket(AF_INET, SOCK_DGRAM)
        recvSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        recvSocket.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        recvSocket.bind(('', 4321))
        while True:
            recvData, addr = recvSocket.recvfrom(1024)
            data = struct.unpack("8f", recvData)
            if data != None:
                recvSocket.close()
                break
        a = data[1:4]
        b = State(a)
        print("获得小车位置")
        return np.array([[b.xyz]])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        car_state = self.receive()
        car_state = car_state[0]
        car_state = car_state[0]
        self.start = car_state
        #print(2)
        car_state = np.hstack([car_state, np.array([0,10])])
        return car_state

    def step(self, action):
        [command, vel] = self.rule(action)
        #print('command', command)
        #print('vel', vel)
        while True:
            server = ServerProxy("http://192.168.1.135:6677")  # 初始化服务器
            # 接受到DQN学习到的动作
            if command == "up":
                server.t_up(vel, 0.5)  # 调用函数并传参
                break
                print('up')
            elif command == "left":
                server.t_left(vel, 0.5)  # 调用函数并传参
                break
                print('left')
            elif command == "right":
                server.t_right(vel, 0.5)
                break
                print('right')
            elif command == "left_up":
                server.t_left_up(vel, vel * 1.5, 0.5)
                break
                print('left_up')
            elif command == "right_up":
                server.t_right_up(vel * 1.5, vel, 0.5)
                break
                print('right_up')

            # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # client.connect(('192.168.1.135', 6688))
            # to_serverdata = input(command).strip().encode('utf-8')
            # client.sendall(to_serverdata)
            # client.close()
            #server.close()

        #car_state = np.hstack([self.receive(), action])
        now_state = self.receive()
        now_state=now_state[0]
        now_state=now_state[0]

        #print(now_state)
        #print(action)

        car_state = np.hstack([now_state, action])
        print(car_state)

        car_action = action


        reward = self.reward_fn(car_state)
        #print('reward = ', reward)
        done = 0.0

        # 成功标志
        dis_end_p = get_ds(car_state[:3], self.end)



        # 碰撞标志
        dis_left_p = get_dis(car_state[:3], self.lane_bound[0], self.lane_bound[1])
        dis_right_p = get_dis(car_state[:3], self.lane_bound[2], self.lane_bound[3])

        print('dis_left_p', dis_left_p)
        print('dis_right_p', dis_right_p)
        print('dis_end_p', dis_end_p)

        # distance from end point or distance from the boundary or angle
        if (dis_left_p < 0.01) or (dis_right_p > 0.2):
            done = 1.0
            reward = -abs(reward * 2)
            print("碰撞左边，重新开始")

        if (dis_left_p > 0.2) or (dis_right_p < 0.01):
            done = 1.0
            reward = -abs(reward * 2)
            print("碰撞右边，重新开始")

        flag = '0'
        if (dis_end_p < 0.05) :
            done = 1.0
            reward = abs(reward * 2)
            flag = '1'
            print("到达终点")



        # other optional return
        #print('car_state',car_state)
        #print('reward', reward)
        #print('done', done)
        return car_state, reward, done, flag

    def reward_fn(self, state):

        # 与终点距离
        dis_end_r = get_ds(state[:3], self.end)
        # distance from start point 
        dis_end = (1 - dis_end_r / self.race) * 10


        # 与中间路线是否偏离
        dis_round_r = get_dis(state[:3], self.waypoint[0], self.waypoint[1])
        # distance from the boundary
        dis_bound = (1 - dis_round_r / self.wide) * 5
        #print(3)

        reward_reward = dis_end * dis_bound

        # 与中间路线偏离过大
        if dis_round_r > 0.005:
            reward_a = - abs(reward_reward * 0.2)
        elif dis_round_r > 0.01:
            reward_a = - abs(reward_reward * 0.5)
        elif dis_round_r > 0.03:
            reward_a = - abs(reward_reward)
        elif dis_round_r > 0.08:
            reward_a = - abs(reward_reward *3)
        else:
            reward_a = 0

        # 与初始点距离过近，则减分
        dis_car_state = self.start
        dis_ini_r = get_ds(state[:3], dis_car_state)
        if dis_ini_r < 0.05:
            reward_b = - dis_end * dis_bound * 0.1
        else:
            reward_b = 0


        # angle devition from waypoint
        # angle =

        # velocity

        #if state[4] == 5:
            #reward_c = -0
        #else:
         #   reward_c = 0

        reward = reward_reward + reward_a + reward_b


        # reward 
        #reward = dis_end * dis_bound
        return reward


def flat_vector(A):
    N = A.shape[0]  # 计算有多少个数据点
    centroid_A = np.mean(A, axis=0)  # 计算每个坐标轴的x、y、z轴方向坐标平均值
    AA = A - np.tile(centroid_A, (N, 1))  # 计算每个坐标值与均值之间的插值

    # SVD奇异值分解,最小奇异值对应的奇异向量就是平面的方向
    U, S, Vt = np.linalg.svd(AA)
    a = Vt[2, 0]
    b = Vt[2, 1]
    c = Vt[2, 2]
    n = [a, b, c]
    d = -np.matmul(n, centroid_A)
    return n, d


def get_point(Nom, D, pose):
    # 计算点到平面的距离
    Nom = np.array(Nom)
    an = Nom[0]
    bn = Nom[1]
    cn = Nom[2]
    dn = D
    xn = pose[0]
    yn = pose[1]
    zn = pose[2]

    dis = abs(an * xn + bn * yn + cn * zn + dn) / math.sqrt(math.pow(an, 2) + math.pow(bn, 2) + math.pow(cn, 2))
    # 计算点在平面上的投影
    Dd = math.sqrt(math.pow(an, 2) + math.pow(bn, 2) + math.pow(cn, 2))

    t = math.sqrt(math.pow(dis, 2) / Dd)

    # x = an*t + xn
    # y = bn*t + yn
    # z = cn*t + zn

    x = xn - an * t
    y = yn - bn * t
    z = zn - cn * t
    po = [x, y, z]
    return po


class State:

    def __init__(self, p):
        self.po = p
        self.solid = np.array([[-1.8683, -1.8706, -2.3688],  # 沙盘平面
                               [-3.3041, -1.8046, -4.2509],
                               [-2.9024, -1.7652, -6.6954],
                               [-0.5434, -1.7117, -8.4542],
                               [3.8408, -1.7702, -2.7662]])
        self.nom, self.d = flat_vector(self.solid)
        self.xyz = get_point(self.nom, self.d, self.po)


def get_dis(point, linea, lineb):

    #print('point',point)
    #print('linea',linea)
    #print('lineb',lineb)
    dis = np.linalg.norm(np.cross((point - linea), (point - lineb))) / np.linalg.norm(lineb - linea)
    return dis


def get_ds(p1, p2):
    #p2=p2[0]
    ds = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2))
    return ds

def back_to_origin(env, track):
    new_track = track[::-1]
    #["up", 'left', 'right', 'left_up', 'right_up']
    for i in new_track:
        [command, vel] = env.back_rule(i)
        if command == "down":
            server = ServerProxy("http://192.168.1.135:6677")  # 初始化服务器
            server.t_down(vel, 0.5)  # 调用函数并传参
            #continue
        if command == "right":
            server = ServerProxy("http://192.168.1.135:6677")  # 初始化服务器
            server.t_right(vel, 0.5)  # 调用函数并传参
            #continue
        if command == "right_down":
            server = ServerProxy("http://192.168.1.135:6677")  # 初始化服务器
            server.t_right_down(vel, vel * 1.5, 0.5)  # 调用函数并传参
            #continue
        if command == "left_down":
            server = ServerProxy("http://192.168.1.135:6677")  # 初始化服务器
            server.t_left_down(1.5*vel, vel, 0.5)
            #continue
        if command == "left":
            server = ServerProxy("http://192.168.1.135:6677")  # 初始化服务器
            server.t_left(vel, 0.5)
            #continue