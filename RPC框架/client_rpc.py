# _*_ coding:utf-8 _*_

from xmlrpc.client import ServerProxy

if __name__ == '__main__':
    server = ServerProxy("http://192.168.1.135:6677") # 初始化服务器
       #接受到DQN学习到的动作
    command = 'up'
    velocity = 30
    if command == "up":
        server.t_up(velocity,1) # 调用函数并传参
    elif action == "left":
        server.t_left(velocity,1) # 调用函数并传参
    elif action == "right":
        server.t_right(velocity,1)
    elif action == "left_up":
        server.t_left_up(velocity,velocity*2,1)
    elif action == "right_up":
        server.t_right_up(velocity*2,velocity,1)
    # if done == 0.0
    #     server.close



