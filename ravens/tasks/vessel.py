#!/usr/bin/env python
"""
The UR5 is centered at zone (0,0,0). The 'square' looks like this:

  |     |   xxxx
  |  o  |   xxxx
  |     |   xxxx
  -------

where `o` is the center of the robot, and `x`'s represent the workspace (so
the horizontal axis is x). Then the 'line' has to fill in the top part of the
square. Each edge has length `length` in code.
"""
import os
import time
import numpy as np
import pybullet as p
import math
from ravens.tasks import Task
from ravens import utils


class Vessel(Task):

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.max_steps = 20
        self.metric = 'zone'
        self.primitive = 'pick_place_vessel'

    def reset(self, env):
        '''重置血管
        '''
        self.total_rewards = 0
        self.goal = {'places': {}, 'steps': [{}]}

        # Hyperparameters for the cable and its `num_parts` beads.
        num_parts = 15
        radius = 0.005
        length = 2 * radius * num_parts * np.sqrt(2)  # 总长？

        # TODO self.goal 目标 part_id -- 某个空间位置

        # 布置桌面上的方形
        # The square -- really 3 edges of it, since .urdf doesn't include a
        # 4th. The square_pose describes the pose relative to a coordinate
        # frame with (0,0,0) at the base of the UR5 robot. Replace the
        # dimension and lengths with desired values in a new .urdf.
        # square_size = (length, length, 0)
        # square_pose = self.random_pose(env, square_size)  # 获得 position, rotation ()
        # print('square_pose: ', square_pose)
        # square_template = 'assets/square/square-template.urdf'
        # replace = {'DIM': (length,), 'HALF': (length / 2 - 0.005,)}
        # urdf = self.fill_template(square_template, replace)  # 把 DIM HALF 替换成具体参数
        # env.add_object(urdf, square_pose, fixed=True)
        # os.remove(urdf)  # 删除临时文件

        # 第四条边作为目标
        # Add goal line, to the missing square edge, enforced via the
        # application of square_pose on zone_position. The zone pose takes on
        # the square pose, to keep it rotationally consistent. The position
        # has y=length/2 because it needs to fill the top part of the square
        # (see diagram above in documentation), and x=0 because we later vary x
        # from -length/2 to length/2 when creating beads in the for loop. Use
        # zone_size for reward function computation, allowing a range of 0.03
        # in the y direction [each bead has diameter 0.01].
        # line_template = 'assets/line/line-template.urdf'
        self.zone_size = (length, 0.03, 0.2)
        zone_range = (self.zone_size[0], self.zone_size[1], 0.001)
        # zone_position = (0, length / 2, 0.001)
        # zone_position = self.apply(square_pose, zone_position)
        # self.zone_pose = (zone_position, square_pose[1])

        # Andy has this commented out. It's nice to see but not needed.
        #urdf = self.fill_template(line_template, {'DIM': (length,)})
        #env.add_object(urdf, self.zone_pose, fixed=True)
        #os.remove(urdf)

        # Add vessels
        self.object_points = {}
        # position = np.float32((0.5,0,0))
        # position, _ = self.random_pose(env, zone_range)  # 随机位置（注意是tuple）

        def add_vessel(position, direction):
            '''position 起始位置, direction 延伸的方向
            '''
            assert np.linalg.norm(direction) == 1, "direction should be unit vector"
            # 定义每个珠子的属性
            # distance = length / num_parts

            for i in range(num_parts):
                if i < num_parts - 1:
                    distance = 0.011  # 珠子之间的距离
                    part_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.01, height=0.01)
                    part_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.01, length=0.01)
                    orientation = p.getQuaternionFromEuler((0, 0.5*math.pi, 0))
                else:
                    distance = 0.055  # 珠子之间的距离
                    part_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.01, height=0.10)
                    part_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.01, length=0.10)  # 主要改形状
                    orientation = p.getQuaternionFromEuler((0, 0.5*math.pi, 0))
                position += [d * distance for d in direction]  # 每次增加一个珠子的距离
                part_id = p.createMultiBody(
                    0.1, part_shape, part_visual, basePosition=position, baseOrientation=orientation)
                # if len(env.objects) > 0:
                if i == 0:  # TODO 让第一个珠子和某个固定悬空物体相连
                    pass
                if i > 0:
                    constraint_id = p.createConstraint(
                        parentBodyUniqueId=env.objects[-1],  # 和前一个珠子相连
                        parentLinkIndex=-1,
                        childBodyUniqueId=part_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_POINT2POINT,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=(0, 0, distance * direction[0]),  # TODO 处理不了x轴之外的方向
                        childFramePosition=(0, 0, 0))
                    p.changeConstraint(constraint_id, maxForce=100)
                # if (i > 0) and (i < num_parts - 1):  # 一头一尾不变色 中间红色
                if i > 0:
                    color = utils.COLORS['red'] + [1]
                    p.changeVisualShape(part_id, -1, rgbaColor=color)
                env.objects.append(part_id)
                # print("part_id:", env.objects)
            return part_id

        utils.cprint('Adding vessel1...', 'green')
        last_part_1 = add_vessel(np.float32((0.2, 0, 0)), [1, 0, 0])  # add vessel 1
        utils.cprint('Adding vessel2...', 'green')
        last_part_2 = add_vessel(np.float32((0.7, 0, 0)), [-1, 0, 0])  # add vessel 2

        true_position = (0.5, 0, 0)
        self.goal['places'][last_part_1] = (true_position, (0, 0, 0, 1.))
        self.goal['places'][last_part_2] = (true_position, (0, 0, 0, 1.))

            # To get target positions for each cable, we need initial reference
            # position `true_position`. Center at x=0 by subtracting length/2.
            # This produces a sequence of points like: {(-a,0,0), ..., (0,0,0),
            # ..., (a,0,0)}. Then apply zone_pose to re-assign `true_position`.
            # No need for orientation target values as beads are symmetric.
            # self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)

            # # Only the second-to-last node is operable.
            # if i == num_parts - 2:
            #     true_position = (radius + distance * i - length / 2, 0, 0)
            #     true_position = self.apply(self.zone_pose, true_position)
            #     # true_position = self.apply(self.zone_pose, (0.2, 0.2, 0))
            #     self.goal['places'][part_id] = (true_position, (0, 0, 0, 1.))
            #     symmetry = 0  # zone-evaluation: symmetry does not matter
            #     self.goal['steps'][0][part_id] = (symmetry, [part_id])

        # Wait for beaded cable to settle.
        env.start()
        time.sleep(1)
        env.pause()
