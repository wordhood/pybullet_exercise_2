import pybullet as p
import pybullet_data
import time
import os
import pybullet_object_models
import random
import numpy as np 

from termcolor import cprint
from pybullet_object_models import ycb_objects

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

KUKA_START_POS = [-1.062, -0.529, -0.843, 1.058, 0.875, -0.595, -1.382, -0.15, 0., 0., 0., 0.15, 0., 0., 0., 0., 0., 0., 0., 1.571, 0., 0., 0.]

O_X_PG = np.array([ [-0.30128315, -0.17094008, -0.93808739,  0.16700348],
                    [-0.74595362,  0.65505938,  0.12020986,  0.00088091],
                    [ 0.59395426,  0.73598689, -0.32487172,  0.18182704],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]
                  ])
W_X_P = np.array([  [ 0.48611527, -0.37474405, -0.78946745, -0.71594822],
                    [-0.48752695,  0.63345039, -0.60088109, -0.70437711],
                    [ 0.72526507,  0.67698413,  0.12523208,  1.21061897],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]
                ])

base_x = [0, 0.225, 0, -0.225]
base_y = [0.225, 0, -0.225, 0]

class PickAndPlace:
    def __init__(self):
        physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        planId = p.loadURDF("plane.urdf")

        self.init_cube_position =  [ 0, -0.75, 0.35]
        self.init_table_position = [ 0,     0,    0]
        self.init_robot_position = [ 0,  -0.8,  0.7]

        
        base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.025])
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.025], rgbaColor=[0.5, 0.3, 0.1, 1])
        

        side_shapes = []
        side_visuals = []
        link_positions = []
        link_orientations = []
        for i in range(4):
            side_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.25, 0.3])
            side_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.25, 0.3], rgbaColor=[0.5, 0.3, 0.1, 1])
            side_shapes.append(side_shape)
            side_visuals.append(side_visual)
            link_positions.append([base_x[i], base_y[i], 0.3])
            link_orientations.append(p.getQuaternionFromEuler([0, 0, 1.57 if i % 2 == 0 else 0]))

        
        basket_body = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=base_shape,
                                        baseVisualShapeIndex=base_visual,
                                        basePosition=[-0.8, -0.95, 0],
                                        linkMasses=[0.1] * 4,  # 每个侧面的质量
                                        linkCollisionShapeIndices=side_shapes,
                                        linkVisualShapeIndices=side_visuals,
                                        linkPositions=link_positions,
                                        linkOrientations=link_orientations,
                                        linkInertialFramePositions=[[0, 0, 0]] * 4,
                                        linkInertialFrameOrientations=[p.getQuaternionFromEuler([0, 0, 0])] * 4,
                                        linkParentIndices=[0] * 4,  # 所有侧面都连接到基座
                                        linkJointTypes=[p.JOINT_FIXED] * 4,  # 固定连接
                                        linkJointAxis=[[0, 0, 0]] * 4)

        self.init_cube_scale = 0.75
        self.init_table_scale = 1.3
        self.init_robot_scale = 1.0
 
        self.ee_link_name = "iiwa7_link_7"

        self.object_names = ['YcbFoamBrick']
        self.drop_gap = 50
        self.work_space = np.array([
                [-0.25, 0.25],
                [-0.1 , 0.25],
                [ 0.83, 0.83],
            ])
        self.dt = 1/240

        self.add_debug_line = True
        self.update_dabug_line_freq = 10
        self.obj_debug_axis_frame = None
        self.ee_debug_axis_frame = None

        self.add_arm_joint_params =False


        self._create_envs()

        if self.add_arm_joint_params:
            self.debug_params = self._add_arm_joint_params()

    def _add_arm_joint_params(self):
        debugParams = []
        for i in range(self.num_robot_joints):
            jointInfo = p.getJointInfo(self.robotId, i)
            jointName = jointInfo[1].decode("utf-8")  # 获取关节名称
            lowerLimit = jointInfo[8]  # 关节下限
            upperLimit = jointInfo[9]  # 关节上限
            paramId = p.addUserDebugParameter(jointName, lowerLimit, upperLimit, 0)  # 使用 URDF 中的上下限
            debugParams.append(paramId)
        return debugParams


    def _create_envs(self):
        self.cubeId = p.loadURDF("cube.urdf", globalScaling=self.init_cube_scale, basePosition=self.init_cube_position, useFixedBase=True)
        self.tableId = p.loadURDF("table/table.urdf", basePosition=self.init_table_position, globalScaling=self.init_table_scale)
        self.robotId = p.loadURDF("/home/linux/Desktop/project/Isaacgym-urdf/urdf/kuka_allegro_description/kuka_allegro.urdf", globalScaling=self.init_robot_scale,  basePosition=self.init_robot_position, useFixedBase=True, flags=p.URDF_MERGE_FIXED_LINKS)
        self.obj_id = None

        self.num_robot_joints = p.getNumJoints(self.robotId)
        # cprint(f"Number of Joints:{self.num_robot_joints}", "blue")

        endEffectorLinkIndex = -1
        for i in range(self.num_robot_joints):
            jointInfo = p.getJointInfo(self.robotId, i)
        # cprint(f"jointInfo: {jointInfo}", "light_magenta")
            if self.ee_link_name in str(jointInfo[12]):
                endEffectorLinkIndex = i
                break
        if endEffectorLinkIndex == -1:
            raise ValueError(f"{self.ee_link_name} not found in robot links.")
        self.ee_index = endEffectorLinkIndex

    def _add_obj(self, obj_name):
        init_obj_pos = np.random.random(3) * (self.work_space[:, 1] - self.work_space[:, 0]) + self.work_space[:, 0]
        flags = p.URDF_USE_INERTIA_FROM_FILE
        obj_id = p.loadURDF(os.path.join(ycb_objects.getDataPath(), obj_name, "model.urdf"), 
                            
                            init_obj_pos, 
                            [0, 0, -0.5736, 0.8192],
                            flags=flags)
        p.changeDynamics(obj_id, -1, lateralFriction=1.0, spinningFriction=0.5, rollingFriction=0.2)

        return obj_id
    
    def reset(self):
        cprint(f">>> reset env ...", "red", "on_black")
        for i in range(self.num_robot_joints):
            p.resetJointState(self.robotId, i, targetValue=KUKA_START_POS[i])  

        self.num_step = 0

    def _move_robot_dof(self, target_dof):
        for j in range(self.num_robot_joints):
            p.setJointMotorControl2(
                self.robotId,
                j,
                p.POSITION_CONTROL,
                targetPosition=target_dof[j],
                force=500,
                positionGain=0.03,
                velocityGain=1
            )

    def _set_robot_dof(self, target_dof):
        for i in range(self.num_robot_joints):
            p.resetJointState(self.robotId, i, targetValue=target_dof[i])  
    
    def _get_X(self, pos, orn):
        rot_max = R.from_quat(orn).as_matrix()
        new_X = np.eye(4)
        new_X[:3, :3] = rot_max
        new_X[:3, 3] = pos

        return new_X
    
    def _X2pose(self, X):
        quat = R.from_matrix(X[:3, :3]).as_quat()
        return X[:3, 3], quat

    def get_link_pose(self, query_index):
        linkState = p.getLinkState(self.robotId, query_index)
        return linkState[0], linkState[1]

    def get_actor_pose(self, actor_index):
        return p.getBasePositionAndOrientation(actor_index)

    def _move_robot_ee_pose(self, target_ee_pos, target_ee_orn):
        jointPoses = p.calculateInverseKinematics(self.robotId, self.ee_index, target_ee_pos, target_ee_orn)
        for j in range(self.num_robot_joints):
            p.setJointMotorControl2(
                self.robotId,
                j,
                p.POSITION_CONTROL,
                targetPosition=jointPoses[j],
                force=500,
                positionGain=0.03,
                velocityGain=1
            )

    def open_hand(self):
        gripper_ids = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        for j in gripper_ids:
            p.setJointMotorControl2(
                self.robotId,
                j,
                p.POSITION_CONTROL,
                targetPosition=0.4,
                force=500,
                positionGain=0.03,
                velocityGain=1
            )
    
    def close_hand(self):
        gripper_ids = [  9, 10, 13, 14, 17, 18, 21, 22]
        for j in gripper_ids:
            p.setJointMotorControl2(
                self.robotId,
                j,
                p.POSITION_CONTROL,
                targetPosition=1.2,
                force=500,
                positionGain=0.01,
                velocityGain=1
            )

    def _step(self):
        # drop object
        if self.num_step % self.drop_gap == 0 and (self.num_step // self.drop_gap) < len(self.object_names):
            self.obj_id = self._add_obj(self.object_names[self.num_step // self.drop_gap])
        
        if self.add_debug_line:
            # add debug line to object
            if self.obj_id is not None:
                objPos, objOrn = self.get_actor_pose(self.obj_id)
                self.obj_debug_axis_frame = self._add_axis_frame(objPos, objOrn, frame_axis_tuple=self.obj_debug_axis_frame)

            # add debug line to ee
            eePos, eeOrn = self.get_link_pose(self.ee_index)
            self.ee_debug_axis_frame = self._add_axis_frame(eePos, eeOrn, frame_axis_tuple=self.ee_debug_axis_frame, axisLength=0.2)

        objPos1, objOrn1 = self.get_actor_pose(self.obj_id)
        W_X_O = self._get_X(objPos1, objOrn1)
        W_X_PG = W_X_O @ O_X_PG

        num_step_0 = 100
        num_step_1 = num_step_0 + 320
        num_step_2 = num_step_1 + 150
        num_step_3 = num_step_2 + 350
        num_step_4 = num_step_3 + 500
        num_step_5 = num_step_4 + 200

        if self.num_step < num_step_1:
            self._move_robot_dof(KUKA_START_POS)

        elif self.num_step < num_step_1:
            self._move_robot_ee_pose(*self._X2pose(W_X_PG))
        
        elif self.num_step < num_step_2 and self.num_step >= num_step_1:
            PG_X_G = np.eye(4)
            PG_X_G[2, 3] = 0
            W_X_G = W_X_PG @ PG_X_G

            self._move_robot_ee_pose(*self._X2pose(W_X_G))
        
        elif self.num_step < num_step_3 and self.num_step >= num_step_2:
            self.close_hand()
            self.grasp_pos, self.grasp_orn = self.get_link_pose(self.ee_index)
        
        elif self.num_step < num_step_4 and self.num_step >= num_step_3:
            # 将1500到2000步之间的操作分为10步
            # 将1500到2000步之间的操作分为10步
            step_size = 5  # 每个阶段50步
            current_stage = (self.num_step - num_step_3) // step_size  # 当前阶段（0到9）
            total_stages = (num_step_4 - num_step_3) // step_size

            # 计算目标位置和旋转的插值
            start_pos, start_orn = self.grasp_pos, self.grasp_orn
            end_pos, end_orn = self._X2pose(W_X_P)
            
            # 线性插值位置
            alpha = (current_stage + 1) / total_stages  # 插值比例
            target_pos = start_pos + alpha * (end_pos - start_pos)

            # 使用 Slerp 插值旋转
            start_quat = R.from_quat(start_orn)
            end_quat = R.from_quat(end_orn)
            
            # 创建 Slerp 插值器
            slerp = Slerp([0, 1], R.concatenate([start_quat, end_quat]))
            target_quat = slerp(alpha)  # 插值
            target_orn = target_quat.as_quat()

            # 移动到插值后的目标位置和旋转
            self._move_robot_ee_pose(target_pos, target_orn)
        elif self.num_step < num_step_5 and self.num_step >= num_step_4:
            self.open_hand()
        else:
            self.obj_id = self._add_obj(self.object_names[0])
            self.num_step = 0

        if self.add_arm_joint_params:
            target_pos = []
            for i in range(self.num_robot_joints):
                target_pos.append(p.readUserDebugParameter(self.debug_params[i])) 
            self._set_robot_dof(target_pos)

        # step simulation
        p.stepSimulation()
        time.sleep(self.dt)

        # update 
        self.num_step += 1

    def _add_axis_frame(self, targetPos, targetOrn, axisLength=0.1, frame_axis_tuple=None):
       
        matrix = p.getMatrixFromQuaternion(targetOrn)
        rot_matrix = np.array(matrix).reshape(3, 3)
        
        
        x_axis = rot_matrix[:, 0]
        y_axis = rot_matrix[:, 1]
        z_axis = rot_matrix[:, 2]
        
       
        x_end = np.array(targetPos) + axisLength * x_axis
        y_end = np.array(targetPos) + axisLength * y_axis
        z_end = np.array(targetPos) + axisLength * z_axis
        
       
        frame_x_axis = p.addUserDebugLine(targetPos, x_end.tolist(), [1, 0, 0], 2, 
                                        replaceItemUniqueId=frame_axis_tuple[0] if frame_axis_tuple is not None else -1)
        frame_y_axis = p.addUserDebugLine(targetPos, y_end.tolist(), [0, 1, 0], 2, 
                                        replaceItemUniqueId=frame_axis_tuple[1] if frame_axis_tuple is not None else -1)
        frame_z_axis = p.addUserDebugLine(targetPos, z_end.tolist(), [0, 0, 1], 2, 
                                        replaceItemUniqueId=frame_axis_tuple[2] if frame_axis_tuple is not None else -1)
        
        return frame_x_axis, frame_y_axis, frame_z_axis
 

    def run(self):
        self.reset()

        while True:
            self._step()
            

        
    




    