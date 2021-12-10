# -GPPO-
基于GPPO算法的树莓派小车路径规划

项目中的文件solid_dqn_fang为算法文件
RPC部分是树莓派小车的控制驱动部分，以及树莓派小车与算法的交互部分
树莓派小车使用的为树莓派的官方桌面系统，树莓派官方网站
https://www.raspberrypi.org/

这个项目是使用GPPO算法来规划小车路径，目的是验证算法在实物小车上的可行性，目前做完的工作是实物小车直接上算法进行训练，但是训练结果一般，需要修改
后续的工作是使用虚拟场景进行预先训练，然后迁移到实物小车，目前使用了UNITY平台作为虚拟场景搭建，但是效果不是很好
后续打算使用ROS作为仿真平台

实现的效果如下
<iframe src="//player.bilibili.com/player.html?bvid=BV1cr4y1D7p4&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

