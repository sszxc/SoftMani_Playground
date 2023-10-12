#!/usr/bin/env python

import os
import sys
import time
import threading
import pkg_resources
import math
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from ravens.gripper import Gripper, Suction
from ravens import tasks, utils
from ravens import Environment


class DualArmEnvironment(Environment):
    def __init__(self, disp=False, hz=240):
        super().__init__(disp, hz)  # 基础的 bullet 环境
        self.primitives["pick_place_vessel"] = self.pick_place_vessel

    def reset(self, task, last_info=None, disable_render_load=True):
        '''初始化双机械臂环境
        '''
        disable_render_load = False  # 初始化的时候允许渲染（会卡顿）
        self.pause()
        self.task = task
        self.objects = []
        self.fixed_objects = []
        if self.use_new_deformable:  # default True
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        else:
            p.resetSimulation()

        p.setGravity(0, 0, -1)  # 手动修改重力
        # p.setGravity(0, 0, -9.8)

        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=20,
            cameraPitch=-35,
            cameraTargetPosition=(0.5, 0, 0),)  # 调整相机位置  TODO 好像不能调整焦距

        # Slightly increase default movej timeout for the more demanding tasks.
        if self.is_bag_env():
            self.t_lim = 60
            if isinstance(self.task, tasks.names['bag-color-goal']):
                self.t_lim = 120

        # Empirically, this seems to make loading URDFs faster w/remote displays.
        if disable_render_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        id_plane = p.loadURDF('assets/plane/plane.urdf', [0, 0, -0.001])
        id_ws = p.loadURDF('assets/ur5/workspace.urdf', [0.5, 0, 0])

        # Load UR5 robot arm equipped with task-specific end effector.
        self.ur5 = p.loadURDF(f'assets/ur5/ur5-{self.task.ee}.urdf', basePosition=(0,0,0))
        ori_co = p.getQuaternionFromEuler((0, 0, math.pi))
        self.ur5_2 = p.loadURDF(f'assets/ur5/ur5-{self.task.ee}.urdf', basePosition=(1.1,0,0), baseOrientation=ori_co)
        self.ee_tip_link_2 = 12
        self.ee_tip_link = 12
        if self.task.ee == 'suction':
            self.ee = Suction(self.ur5, 11)
            self.ee_2 = Suction(self.ur5_2, 11)
        elif self.task.ee == 'gripper':
            self.ee = Robotiq2F85(self.ur5, 9)
            self.ee_tip_link = 10
        else:
            self.ee = Gripper()

        # Get revolute joint indices of robot (skip fixed joints).
        utils.cprint('UR5 Arm1 setup...', 'blue')
        num_joints = p.getNumJoints(self.ur5)
        # print("num_joints:", num_joints)
        joints = [p.getJointInfo(self.ur5, i) for i in range(num_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]
        # print('self.joints:\n', '\n'.join([str(j) for j in joints]))

        utils.cprint('UR5 Arm2 setup...', 'blue')
        num_joints_2 = p.getNumJoints(self.ur5_2)
        # print("num_joints2:", num_joints_2)
        joints_2 = [p.getJointInfo(self.ur5_2, i) for i in range(num_joints_2)]
        self.joints_2 = [j[0] for j in joints_2 if j[2] == p.JOINT_REVOLUTE]
        # print('self.joints2:\n', '\n'.join([str(j) for j in joints_2]))

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.homej[i])
            p.resetJointState(self.ur5_2, self.joints_2[i], self.homej[i])

        # Get end effector tip pose in home configuration.
        ee_tip_state = p.getLinkState(self.ur5, self.ee_tip_link)
        self.home_pose = np.array(ee_tip_state[0] + ee_tip_state[1])
        
        ee_tip_state_2 = p.getLinkState(self.ur5_2, self.ee_tip_link_2)
        self.home_pose_2 = np.array(ee_tip_state_2[0] + ee_tip_state_2[1])

        # Reset end effector.
        self.ee.release()
        self.ur5_list = [self.ur5, self.ee_tip_link, self.joints, self.ee]
        self.ee_2.release()
        self.ur5_2_list = [self.ur5_2, self.ee_tip_link_2, self.joints_2, self.ee_2]
        

        # Seems like this should be BEFORE reset()
        # since for bag-items we may assign to True!
        task.exit_gracefully = False

        # Reset task. 重置血管
        if last_info is not None:
            task.reset(self, last_info)
        else:
            task.reset(self)

        # Daniel: might be useful to have this debugging tracker.
        self.IDTracker = utils.TrackIDs()
        self.IDTracker.add(id_plane, 'Plane')
        self.IDTracker.add(id_ws, 'Workspace')
        self.IDTracker.add(self.ur5, 'UR5')
        try:
            self.IDTracker.add(self.ee.body, 'Gripper.body')
        except:
            pass

        # Daniel: add other IDs, but not all envs use the ID tracker.
        try:
            task_IDs = task.get_ID_tracker()
            for i in task_IDs:
                self.IDTracker.add(i, task_IDs[i])
        except AttributeError:
            pass
        #print(self.IDTracker)  # If doing multiple episodes, check if I reset the ID dict!
        assert id_ws == 1, f'Workspace ID: {id_ws}'

        # Daniel: tune gripper for deformables if applicable, and CHECK HZ!!
        if self.is_softbody_env():  # default False
            self.ee.set_def_threshold(threshold=self.task.def_threshold)
            self.ee.set_def_nb_anchors(nb_anchors=self.task.def_nb_anchors)
            assert self.hz >= 480, f'Error, hz={self.hz} is too small!'

        # Restart simulation.
        self.start()
        if disable_render_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        (obs, _, _, _) = self.step()
        return obs
    
    def movep(self, arm1_pose, arm2_pose, speed=0.01):
        """Move dual UR5s to target end effector pose."""
        # # Keep joint angles between -180/+180
        # targj[5] = ((targj[5] + np.pi) % (2 * np.pi) - np.pi)
        if arm1_pose is None:
            arm1_targj = None
        else:
            arm1_targj = self.solve_IK(self.ur5_list, arm1_pose)
            
        if arm2_pose is None:
            arm2_targj=None
        else:
            arm2_targj = self.solve_IK(self.ur5_2_list, arm2_pose)
            
        return self.movej(arm1_targj, arm2_targj, speed, self.t_lim)

    def solve_IK(self, arm, pose):
        '''add parameter 'arm'
        '''
        homej_list = np.array(self.homej).tolist()
        joints = p.calculateInverseKinematics(
            bodyUniqueId=arm[0],
            endEffectorLinkIndex=arm[1],
            targetPosition=pose[:3],
            targetOrientation=pose[3:],
            lowerLimits=[-17, -2.3562, -17, -17, -17, -17],
            upperLimits=[17, 0, 17, 17, 17, 17],
            jointRanges=[17] * 6,
            restPoses=homej_list,
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.array(joints)
        joints[joints > 2 * np.pi] = joints[joints > 2 * np.pi] - 2 * np.pi
        joints[joints < -2 * np.pi] = joints[joints < -2 * np.pi] + 2 * np.pi
        return joints
    
    def movej(self, arm1_targj, arm2_targj, speed=0.01, t_lim=20):
        """Move dual UR5s to target joint configuration."""
        t0 = time.time()
        flag1 = False
        flag2 = False
        while (time.time() - t0) < t_lim:
            if arm1_targj is not None:
                arm1_currj = [p.getJointState(self.ur5_list[0], i)[0] for i in self.ur5_list[2]]
                arm1_currj = np.array(arm1_currj)
                arm1_diffj = arm1_targj - arm1_currj

                arm1_norm = np.linalg.norm(arm1_diffj)
                arm1_v = arm1_diffj / arm1_norm if arm1_norm > 0 else 0
                arm1_stepj = arm1_currj + arm1_v * speed
                arm1_gains = np.ones(len(self.ur5_list[2]))
                p.setJointMotorControlArray(
                    bodyIndex=self.ur5_list[0],
                    jointIndices=self.ur5_list[2],
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=arm1_stepj,
                    positionGains=arm1_gains)      
                
                if all(np.abs(arm1_diffj) < 1e-2):
                    flag1 = True
            else:
                flag1 = True
                
            if arm2_targj is not None:                       
                arm2_currj = [p.getJointState(self.ur5_2_list[0], i)[0] for i in self.ur5_2_list[2]]
                arm2_currj = np.array(arm2_currj)
                arm2_diffj = arm2_targj - arm2_currj
                
                arm2_norm = np.linalg.norm(arm2_diffj)
                arm2_v = arm2_diffj / arm2_norm if arm2_norm > 0 else 0
                arm2_stepj = arm2_currj + arm2_v * speed
                arm2_gains = np.ones(len(self.ur5_list[2]))
                p.setJointMotorControlArray(
                    bodyIndex=self.ur5_2_list[0],
                    jointIndices=self.ur5_2_list[2],
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=arm2_stepj,
                    positionGains=arm2_gains)
                
                if all(np.abs(arm2_diffj) < 1e-2):
                        flag2 = True
            else:
                flag2 = True
            time.sleep(0.005)
            if flag1 and flag2:
                return True
        print('Warning: movej exceeded {} sec timeout. Skipping.'.format(t_lim))
        return False
    
    
    def pick_place_vessel(self, arm1_pose0, arm1_pose1, arm2_pose0, arm2_pose1):
        """Execute pick and place primitive.

        Standard ravens tasks use the `delta` vector to lower the gripper
        until it makes contact with something. With deformables, however, we
        need to consider cases when the gripper could detect a rigid OR a
        soft body (cloth or bag); it should grip the first item it touches.
        This is handled in the Gripper class.

        Different deformable ravens tasks use slightly different parameters
        for better physics (and in some cases, faster simulation). Therefore,
        rather than make special cases here, those tasks will define their
        own action parameters, which we use here if they exist. Otherwise, we
        stick to defaults from standard ravens. Possible action parameters a
        task might adjust:

            speed: how fast the gripper moves.
            delta_z: how fast the gripper lowers for picking / placing.
            prepick_z: height of the gripper when it goes above the target
                pose for picking, just before it lowers.
            postpick_z: after suction gripping, raise to this height, should
                generally be low for cables / cloth.
            preplace_z: like prepick_z, but for the placing pose.
            pause_place: add a small pause for some tasks (e.g., bags) for
                slightly better soft body physics.
            final_z: height of the gripper after the action. Recommended to
                leave it at the default of 0.3, because it has to be set high
                enough to avoid the gripper occluding the workspace when
                generating color/depth maps.
        Args:
            pose0: picking pose.
            pose1: placing pose.

        Returns:
            A bool indicating whether the action succeeded or not, via
            checking the sequence of movep calls. If any movep failed, then
            self.step() will terminate the episode after this action.
        """
        print("arm1_pose0, arm1_pose1, arm2_pose0, arm2_pose1:", arm1_pose0, arm1_pose1, arm2_pose0, arm2_pose1)
        # Defaults used in the standard Ravens environments.
        speed = 0.01
        delta_z = -0.001
        prepick_z = 0.3
        postpick_z = 0.3
        preplace_z = 0.3
        pause_place = 0.0
        final_z = 0.3

        # Find parameters, which may depend on the task stage.
        if hasattr(self.task, 'primitive_params'):
            ts = self.task.task_stage
            if 'prepick_z' in self.task.primitive_params[ts]:
                prepick_z = self.task.primitive_params[ts]['prepick_z']
            speed       = self.task.primitive_params[ts]['speed']
            delta_z     = self.task.primitive_params[ts]['delta_z']
            postpick_z  = self.task.primitive_params[ts]['postpick_z']
            preplace_z  = self.task.primitive_params[ts]['preplace_z']
            pause_place = self.task.primitive_params[ts]['pause_place']

        # Used to track deformable IDs, so that we can get the vertices.
        def_IDs = []
        if hasattr(self.task, 'def_IDs'):
            def_IDs = self.task.def_IDs

        # Otherwise, proceed as normal.
        success = True
        arm1_pick_position = np.array(arm1_pose0[0])
        arm1_pick_rotation = np.array(arm1_pose0[1])
        arm1_prepick_position = arm1_pick_position.copy()
        arm1_prepick_position[2] = prepick_z
        
        arm2_pick_position = np.array(arm2_pose0[0])
        arm2_pick_rotation = np.array(arm2_pose0[1])
        arm2_prepick_position = arm2_pick_position.copy()
        arm2_prepick_position[2] = prepick_z

        # Execute picking motion primitive.
        arm1_prepick_pose = np.hstack((arm1_prepick_position, arm1_pick_rotation))
        arm2_prepick_pose = np.hstack((arm2_prepick_position, arm2_pick_rotation))
        
        
        success &= self.movep(arm1_prepick_pose, arm2_prepick_position)
        
        arm1_target_pose = arm1_prepick_pose.copy()
        arm2_target_pose = arm2_prepick_pose.copy()
        delta = np.array([0, 0, delta_z, 0, 0, 0, 0])

        # Lower gripper until (a) touch object (rigid OR softbody), or (b) hit ground.
        while True:
            if not self.ee.detect_contact(def_IDs) and arm1_target_pose[2] > 0:
                arm1_target_pose += delta
            else:
                arm1_target_pose = None
                
            if not self.ee_2.detect_contact(def_IDs) and arm2_target_pose[2] > 0:
                arm2_target_pose += delta
            else:
                arm2_target_pose = None
                
            if arm1_target_pose is None and arm2_target_pose is None:
                break
            else:   
                success &= self.movep(arm1_target_pose, arm2_target_pose)

        # Create constraint (rigid objects) or anchor (deformable).
        self.ee.activate(self.objects, def_IDs)
        self.ee_2.activate(self.objects, def_IDs)


        # Increase z slightly (or hard-code it) and check picking success.
        arm1_prepick_pose[2] = 0.1
        arm2_prepick_pose[2] = 0.1
        success &= self.movep(arm1_prepick_pose, arm2_prepick_pose, speed=0.001)

        pick_success = self.ee.check_grasp()

        if pick_success:
            arm1_place_position = np.array(arm1_pose1[0])
            arm1_place_rotation = np.array(arm1_pose1[1])
            
            arm2_place_position = np.array(arm2_pose1[0])
            arm2_place_rotation = np.array(arm2_pose1[1])
            
            
            arm1_place_pose = np.hstack((arm1_place_position, arm1_place_rotation))
            arm2_place_pose = np.hstack((arm2_place_position, arm2_place_rotation))
            
            
            success &= self.movep(arm1_place_pose, arm2_place_pose, speed=0.001)
            # preplace_position = place_position.copy()
            # preplace_position[2] = 0.3 + pick_position[2]

            # Execute placing motion primitive if pick success.
        #     preplace_pose = np.hstack((preplace_position, place_rotation))
        #     if self.is_softbody_env() or self.is_new_cable_env():
        #         preplace_pose[2] = preplace_z
        #         success &= self.movep(preplace_pose, speed=speed)
        #         time.sleep(pause_place) # extra rest for bags
        #     elif isinstance(self.task, tasks.names['cable']) or \
        #         isinstance(self.task, tasks.names['cable-vessel']):
        #         preplace_pose[2] = 0.03
        #         # preplace_pose[2] = 0.2  # extra target height
        #         success &= self.movep(preplace_pose, speed=0.001)
        #     else:
        #         success &= self.movep(preplace_pose)

        #     # Lower the gripper. Here, we have a fixed speed=0.01. TODO: consider additional
        #     # testing with bags, so that the 'lowering' process for bags is more reliable.
        #     target_pose = preplace_pose.copy()
        #     while not self.ee.detect_contact(def_IDs) and target_pose[2] > 0:
        #         target_pose += delta
        #         success &= self.movep(target_pose)

        #     # Release AND get gripper high up, to clear the view for images.
        #     self.ee.release()
        #     preplace_pose[2] = final_z
        #     success &= self.movep(preplace_pose)
        # else:
        #     # Release AND get gripper high up, to clear the view for images.
        #     self.ee.release()
        #     prepick_pose[2] = final_z
        #     success &= self.movep(prepick_pose)
        # return success
