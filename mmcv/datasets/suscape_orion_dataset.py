# ------------------------------------------------------------------------
# SUScape Dataset for ORION Open-Loop Evaluation
# Copyright (c) Xiaomi, Inc. All rights reserved.
# ------------------------------------------------------------------------
"""
SUScape Dataset for ORION Model Evaluation

This dataset handles the SUScape dataset format which includes:
- CSV files for vehicle trajectories (0.csv format)
- Multi-view camera images from scene directories
- No map information available

Data structure:
    suscape_scenes/
    ├── raws/
    │   ├── scene-000000/
    │   │   ├── 0.csv
    │   │   ├── CAM_FRONT/
    │   │   ├── CAM_FRONT_LEFT/
    │   │   ├── CAM_FRONT_RIGHT/
    │   │   ├── CAM_BACK/
    │   │   ├── CAM_BACK_LEFT/
    │   │   └── CAM_BACK_RIGHT/
    │   ├── scene-000001/
    │   └── ...
"""

import copy
import numpy as np
import os
from os import path as osp
import torch
import pandas as pd
import pickle
import mmcv
from mmcv.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmcv.datasets.map_utils.mean_ap import eval_map
from .nuscenes_styled_eval_utils import DetectionMetrics, EvalBoxes, DetectionBox, center_distance
import math


def invert_matrix_egopose_numpy(egopose):
    """Invert egopose matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


@DATASETS.register_module()
class SUScapeOrionDataset(Custom3DDataset):
    """SUScape dataset for ORION model evaluation.
    
    This dataset loads trajectory data from CSV files and camera images
    from scene directories for open-loop evaluation of L2 metrics and
    collision rates.
    
    Args:
        data_root (str): Root directory of SUScape dataset (e.g., 'data/suscape_scenes')
        ann_file (str): Path to annotation file (will be generated if not exists)
        queue_length (int): Number of historical frames
        past_frames (int): Number of past frames for history trajectory
        future_frames (int): Number of future frames for prediction
        point_cloud_range (list): Range of point cloud
        name_mapping (dict): Mapping from object type names to class names
        eval_cfg (dict): Evaluation configuration
        **kwargs: Additional arguments for Custom3DDataset
    """
    
    def __init__(
        self, 
        queue_length=4,
        seq_mode=False,
        seq_split_num=1,
        with_velocity=True,
        sample_interval=1,
        name_mapping=None,
        eval_cfg=None,
        past_frames=2,
        future_frames=6,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        polyline_points_num=20,
        *args,
        eval_mode=['det'],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.with_velocity = with_velocity
        self.NameMapping = name_mapping if name_mapping is not None else {}
        self.eval_cfg = eval_cfg if eval_cfg is not None else {}
        self.sample_interval = sample_interval
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.point_cloud_range = np.array(point_cloud_range)
        self.polyline_points_num = polyline_points_num
        self.eval_mode = eval_mode
        
        # SUScape doesn't have map information
        self.map_element_class = {}
        self.MAPCLASSES = []
        self.NUM_MAPCLASSES = 0
        
        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag()

    def load_annotations(self, ann_file):
        """Load or generate annotations from SUScape dataset.
        
        If ann_file doesn't exist, scan the data_root/raws directory to
        generate annotations from CSV files.
        """
        if osp.exists(ann_file):
            print(f'Loading annotations from {ann_file}')
            return super().load_annotations(ann_file)
        
        print(f'Annotation file {ann_file} not found. Generating from SUScape data...')
        data_infos = self._generate_annotations_from_suscape()
        
        # Save generated annotations
        mmcv.mkdir_or_exist(osp.dirname(ann_file))
        mmcv.dump(data_infos, ann_file)
        print(f'Saved {len(data_infos)} annotations to {ann_file}')
        
        return data_infos
    
    def _generate_annotations_from_suscape(self):
        """Generate annotations by scanning SUScape raws directory and parsing CSV files."""
        raws_dir = osp.join(self.data_root, 'raws')
        if not osp.exists(raws_dir):
            raise FileNotFoundError(f'SUScape raws directory not found: {raws_dir}')
        
        data_infos = []
        scene_dirs = sorted([d for d in os.listdir(raws_dir) if d.startswith('scene-')])
        
        print(f'Found {len(scene_dirs)} scenes in {raws_dir}')
        
        for scene_name in mmcv.track_iter_progress(scene_dirs):
            scene_dir = osp.join(raws_dir, scene_name)
            csv_file = osp.join(scene_dir, '0.csv')
            
            if not osp.exists(csv_file):
                print(f'Warning: CSV file not found in {scene_dir}, skipping...')
                continue
            
            # Parse CSV file
            scene_infos = self._parse_csv_file(csv_file, scene_dir, scene_name)
            data_infos.extend(scene_infos)
        
        return data_infos
    
    def _parse_csv_file(self, csv_file, scene_dir, scene_name):
        """Parse SUScape CSV file to extract frame-by-frame data.
        
        CSV format:
        TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y, V_X, V_Y, A_X, A_Y, YAW, DYAW, DDYAW, CITY_NAME
        
        Returns:
            list of frame info dicts
        """
        df = pd.read_csv(csv_file, sep='\t')
        
        # Group by timestamp to get frames
        grouped = df.groupby('TIMESTAMP')
        timestamps = sorted(df['TIMESTAMP'].unique())
        
        frame_infos = []
        
        for frame_idx, timestamp in enumerate(timestamps):
            frame_data = grouped.get_group(timestamp)
            
            # Get ego vehicle data
            ego_data = frame_data[frame_data['TRACK_ID'] == 'ego']
            if len(ego_data) == 0:
                continue
            
            ego_data = ego_data.iloc[0]
            
            # Get other objects (non-ego)
            objects_data = frame_data[frame_data['TRACK_ID'] != 'ego']
            
            # Build frame info
            frame_info = self._build_frame_info(
                scene_name, scene_dir, frame_idx, timestamp,
                ego_data, objects_data
            )
            
            frame_infos.append(frame_info)
        
        return frame_infos
    
    def _build_frame_info(self, scene_name, scene_dir, frame_idx, timestamp, ego_data, objects_data):
        """Build frame information dictionary."""
        
        # Ego vehicle information
        ego_x, ego_y = ego_data['X'], ego_data['Y']
        ego_vx, ego_vy = ego_data['V_X'], ego_data['V_Y']
        ego_ax, ego_ay = ego_data['A_X'], ego_data['A_Y']
        ego_yaw = ego_data['YAW'] * np.pi / 180  # Convert to radians
        ego_dyaw = ego_data['DYAW'] * np.pi / 180
        
        # Build camera paths (assuming standard camera naming)
        camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                       'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        sensors = {}
        
        # Add LIDAR_TOP placeholder (required by the system)
        # Create transformation matrices
        lidar2ego = np.eye(4)
        world2lidar = np.eye(4)
        world2lidar[:2, 3] = [-ego_x, -ego_y]
        # Rotation matrix for yaw
        cos_yaw, sin_yaw = np.cos(-ego_yaw), np.sin(-ego_yaw)
        world2lidar[:2, :2] = [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]]
        
        sensors['LIDAR_TOP'] = {
            'lidar2ego': lidar2ego,
            'world2lidar': world2lidar,
        }
        
        # Add camera information
        for cam_name in camera_names:
            cam_dir = osp.join(scene_dir, cam_name)
            if not osp.exists(cam_dir):
                continue
            
            # Find image file for this frame
            img_files = sorted([f for f in os.listdir(cam_dir) if f.endswith(('.jpg', '.png'))])
            if frame_idx < len(img_files):
                img_path = osp.join('raws', scene_name, cam_name, img_files[frame_idx])
                
                # Default camera intrinsics (can be adjusted based on actual camera specs)
                intrinsic = self._get_default_intrinsic()
                cam2ego = self._get_default_cam2ego(cam_name)
                
                sensors[cam_name] = {
                    'data_path': img_path,
                    'intrinsic': intrinsic,
                    'cam2ego': cam2ego,
                }
        
        # Parse objects (ground truth boxes)
        gt_boxes = []
        gt_names = []
        gt_ids = []
        npc2world = []
        num_points = []
        
        for _, obj in objects_data.iterrows():
            obj_type = obj['OBJECT_TYPE']
            track_id = obj['TRACK_ID']
            
            # Extract object information
            x, y = obj['X'], obj['Y']
            vx, vy = obj['V_X'], obj['V_Y']
            yaw = obj['YAW'] * np.pi / 180
            
            # Default size based on object type (can be refined)
            if obj_type == 'Vehicle':
                l, w, h = 4.5, 2.0, 1.6
            elif obj_type == 'Pedestrian':
                l, w, h = 0.6, 0.6, 1.7
            elif obj_type == 'Bicycle':
                l, w, h = 1.8, 0.6, 1.5
            else:
                l, w, h = 2.0, 2.0, 1.5
            
            # Bounding box in LiDAR coordinates: [x, y, z, l, w, h, yaw, vx, vy]
            # Transform from world to ego coordinates
            dx, dy = x - ego_x, y - ego_y
            cos_ego, sin_ego = np.cos(-ego_yaw), np.sin(-ego_yaw)
            x_ego = cos_ego * dx - sin_ego * dy
            y_ego = sin_ego * dx + cos_ego * dy
            yaw_ego = yaw - ego_yaw
            
            # Velocity in ego frame
            vx_ego = cos_ego * vx - sin_ego * vy
            vy_ego = sin_ego * vx + cos_ego * vy
            
            gt_box = [x_ego, y_ego, 0.0, l, w, h, yaw_ego, vx_ego, vy_ego]
            gt_boxes.append(gt_box)
            gt_names.append(obj_type)
            gt_ids.append(str(track_id))
            num_points.append(100)  # Placeholder
            
            # npc2world transformation
            npc_transform = np.eye(4)
            npc_transform[:2, 3] = [x, y]
            cos_yaw_obj, sin_yaw_obj = np.cos(yaw), np.sin(yaw)
            npc_transform[:2, :2] = [[cos_yaw_obj, -sin_yaw_obj], [sin_yaw_obj, cos_yaw_obj]]
            npc2world.append(npc_transform)
        
        gt_boxes = np.array(gt_boxes, dtype=np.float32) if len(gt_boxes) > 0 else np.zeros((0, 9), dtype=np.float32)
        
        frame_info = {
            'folder': scene_name,
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'town_name': ego_data['CITY_NAME'],
            'sensors': sensors,
            'ego_translation': np.array([ego_x, ego_y, 0.0], dtype=np.float32),
            'ego_yaw': ego_yaw,
            'ego_vel': np.array([ego_vx, ego_vy, 0.0], dtype=np.float32),
            'ego_accel': np.array([ego_ax, ego_ay, 0.0], dtype=np.float32),
            'ego_rotation_rate': np.array([0.0, 0.0, ego_dyaw], dtype=np.float32),
            'ego_size': np.array([4.084, 1.85, 1.562], dtype=np.float32),  # Default ego size
            'steer': 0.0,  # Not available in CSV
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
            'gt_ids': gt_ids,
            'num_points': np.array(num_points, dtype=np.int32) if len(num_points) > 0 else np.zeros(0, dtype=np.int32),
            'npc2world': npc2world if len(npc2world) > 0 else [],
        }
        
        return frame_info
    
    def _get_default_intrinsic(self):
        """Get default camera intrinsic matrix."""
        # Default camera intrinsic (can be adjusted)
        fx, fy = 640.0, 640.0  # Focal length
        cx, cy = 640.0, 360.0  # Principal point
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return intrinsic
    
    def _get_default_cam2ego(self, cam_name):
        """Get default camera to ego transformation."""
        cam2ego = np.eye(4, dtype=np.float32)
        
        # Default camera positions (adjust based on actual setup)
        if 'FRONT' in cam_name and 'LEFT' not in cam_name and 'RIGHT' not in cam_name:
            cam2ego[:3, 3] = [1.7, 0.0, 1.5]  # Front camera
        elif 'FRONT_LEFT' in cam_name:
            cam2ego[:3, 3] = [1.5, -0.5, 1.5]
            cam2ego[:3, :3] = self._rotation_matrix_y(np.pi / 6)  # 30 degrees left
        elif 'FRONT_RIGHT' in cam_name:
            cam2ego[:3, 3] = [1.5, 0.5, 1.5]
            cam2ego[:3, :3] = self._rotation_matrix_y(-np.pi / 6)  # 30 degrees right
        elif 'BACK' in cam_name and 'LEFT' not in cam_name and 'RIGHT' not in cam_name:
            cam2ego[:3, 3] = [-1.0, 0.0, 1.5]
            cam2ego[:3, :3] = self._rotation_matrix_y(np.pi)  # 180 degrees
        elif 'BACK_LEFT' in cam_name:
            cam2ego[:3, 3] = [-1.0, -0.5, 1.5]
            cam2ego[:3, :3] = self._rotation_matrix_y(5 * np.pi / 6)  # 150 degrees
        elif 'BACK_RIGHT' in cam_name:
            cam2ego[:3, 3] = [-1.0, 0.5, 1.5]
            cam2ego[:3, :3] = self._rotation_matrix_y(-5 * np.pi / 6)  # -150 degrees
        
        return cam2ego
    
    def _rotation_matrix_y(self, angle):
        """Create rotation matrix around Y axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=np.float32)
    
    def _set_sequence_group_flag(self):
        """Set each sequence to be a different group."""
        res = []
        curr_sequence = 0
        curr_scene_token = self.data_infos[0]['folder']
        
        for idx in range(len(self.data_infos)):
            if idx != 0 and self.data_infos[idx]['folder'] != curr_scene_token:
                curr_sequence += 1
                curr_scene_token = self.data_infos[idx]['folder']
            res.append(curr_sequence)
        
        self.flag = np.array(res, dtype=np.int64)
        
        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, bin_counts[curr_flag], 
                                 math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]]
                    )
                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1
                
                assert len(new_flags) == len(self.flag)
                self.flag = np.array(new_flags, dtype=np.int64)
    
    def invert_pose(self, pose):
        """Invert pose matrix."""
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = -inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose
    
    def get_data_info(self, index):
        """Get data info for a specific index."""
        info = self.data_infos[index]
        
        # Apply name mapping
        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]
        
        ego2global = np.linalg.inv(info['sensors']['LIDAR_TOP']['world2lidar'])
        ego_pose = ego2global
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        
        input_dict = dict(
            folder=info['folder'],
            scene_token=info['folder'],
            frame_idx=info['frame_idx'],
            ego_yaw=np.nan_to_num(info['ego_yaw'], nan=np.pi/2),
            ego_translation=info['ego_translation'],
            sensors=info['sensors'],
            ego_pose=ego_pose,
            ego_pose_inv=ego_pose_inv,
            world2lidar=info['sensors']['LIDAR_TOP']['world2lidar'],
            lidar2ego=info['sensors']['LIDAR_TOP']['lidar2ego'],
            gt_ids=info['gt_ids'],
            gt_boxes=info['gt_boxes'],
            gt_names=info['gt_names'],
            ego_vel=info['ego_vel'],
            ego_accel=info['ego_accel'],
            ego_rotation_rate=info['ego_rotation_rate'],
            npc2world=info['npc2world'],
            timestamp=info['frame_idx'] / 10
        )
        
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
            lidar2global = self.invert_pose(info['sensors']['LIDAR_TOP']['world2lidar'])
            
            for sensor_type, cam_info in info['sensors'].items():
                if 'CAM' not in sensor_type:
                    continue
                
                image_paths.append(osp.join(self.data_root, cam_info['data_path']))
                cam2ego = cam_info['cam2ego']
                intrinsic = cam_info['intrinsic']
                intrinsic_pad = np.eye(4)
                intrinsic_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2cam = self.invert_pose(cam2ego) @ lidar2ego
                lidar2img = intrinsic_pad @ lidar2cam
                lidar2img_rts.append(lidar2img)
                cam_intrinsics.append(intrinsic_pad)
                lidar2cam_rts.append(lidar2cam)
            
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    l2g_r_mat=lidar2global[0:3, 0:3],
                    l2g_t=lidar2global[0:3, 3]
                ))
        
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        
        yaw = input_dict['ego_yaw']
        rotation = list(Quaternion(axis=[0, 0, 1], radians=yaw))
        
        if yaw < 0:
            yaw += 2 * np.pi
        yaw_in_degree = yaw / np.pi * 180
        
        can_bus = np.zeros(18)
        can_bus[:3] = input_dict['ego_translation']
        can_bus[3:7] = rotation
        can_bus[7:10] = input_dict['ego_vel']
        can_bus[10:13] = input_dict['ego_accel']
        can_bus[13:16] = input_dict['ego_rotation_rate']
        can_bus[16] = yaw
        can_bus[17] = yaw_in_degree
        input_dict['can_bus'] = can_bus
        
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0:2] = input_dict['ego_translation'][0:2]
        ego_lcf_feat[2:4] = input_dict['ego_accel'][0:2]
        ego_lcf_feat[4] = input_dict['ego_rotation_rate'][-1]
        ego_lcf_feat[5] = info['ego_size'][1]
        ego_lcf_feat[6] = info['ego_size'][0]
        ego_lcf_feat[7] = np.sqrt(input_dict['ego_vel'][0]**2 + input_dict['ego_vel'][1]**2)
        ego_lcf_feat[8] = info['steer']
        
        ego_his_trajs, ego_fut_trajs, ego_fut_masks, command, command_nohot = self.get_ego_trajs(
            index, self.sample_interval, self.past_frames, self.future_frames
        )
        input_dict['ego_his_trajs'] = ego_his_trajs
        input_dict['ego_fut_trajs'] = ego_fut_trajs
        input_dict['ego_fut_masks'] = ego_fut_masks
        input_dict['ego_fut_cmd'] = command
        input_dict['command'] = command_nohot
        input_dict['ego_lcf_feat'] = ego_lcf_feat
        input_dict['fut_valid_flag'] = (ego_fut_masks == 1).all()
        
        return input_dict
    
    def get_ego_trajs(self, index, sample_interval, past_frames, future_frames):
        """Get ego vehicle historical and future trajectories."""
        scene_token = self.data_infos[index]['folder']
        frame_idx = self.data_infos[index]['frame_idx']
        
        # Initialize trajectories
        ego_his_trajs = np.zeros((past_frames, 2), dtype=np.float32)
        ego_fut_trajs = np.zeros((future_frames, 2), dtype=np.float32)
        ego_fut_masks = np.zeros(future_frames, dtype=np.float32)
        
        current_pos = self.data_infos[index]['ego_translation'][:2]
        current_yaw = self.data_infos[index]['ego_yaw']
        
        # Get historical trajectory
        for i in range(past_frames):
            hist_idx = index - (past_frames - i) * sample_interval
            if hist_idx >= 0 and self.data_infos[hist_idx]['folder'] == scene_token:
                hist_pos = self.data_infos[hist_idx]['ego_translation'][:2]
                # Transform to ego frame
                delta = hist_pos - current_pos
                cos_yaw, sin_yaw = np.cos(-current_yaw), np.sin(-current_yaw)
                ego_his_trajs[i] = [
                    cos_yaw * delta[0] - sin_yaw * delta[1],
                    sin_yaw * delta[0] + cos_yaw * delta[1]
                ]
        
        # Get future trajectory
        for i in range(future_frames):
            fut_idx = index + (i + 1) * sample_interval
            if fut_idx < len(self.data_infos) and self.data_infos[fut_idx]['folder'] == scene_token:
                fut_pos = self.data_infos[fut_idx]['ego_translation'][:2]
                # Transform to ego frame
                delta = fut_pos - current_pos
                cos_yaw, sin_yaw = np.cos(-current_yaw), np.sin(-current_yaw)
                ego_fut_trajs[i] = [
                    cos_yaw * delta[0] - sin_yaw * delta[1],
                    sin_yaw * delta[0] + cos_yaw * delta[1]
                ]
                ego_fut_masks[i] = 1.0
        
        # Command (simplified - can be enhanced based on trajectory)
        command = np.zeros(6, dtype=np.float32)
        command[0] = 1.0  # Default: go straight
        command_nohot = 0
        
        return ego_his_trajs, ego_fut_trajs, ego_fut_masks, command, command_nohot
    
    def get_ann_info(self, index):
        """Get annotation information."""
        info = self.data_infos[index]
        gt_boxes = info['gt_boxes']
        gt_names = info['gt_names']
        gt_ids = info['gt_ids']
        
        gt_bboxes_3d = gt_boxes[:, :7] if len(gt_boxes) > 0 else np.zeros((0, 7), dtype=np.float32)
        gt_velocities = gt_boxes[:, 7:9] if len(gt_boxes) > 0 else np.zeros((0, 2), dtype=np.float32)
        
        gt_labels_3d = np.array([self.cat2id.get(name, -1) for name in gt_names], dtype=np.int64)
        
        # Get agent features for future trajectories (simplified)
        num_agents = len(gt_boxes)
        agent_fut_trajs = np.zeros((num_agents, self.future_frames * 2), dtype=np.float32)
        agent_fut_masks = np.zeros((num_agents, self.future_frames), dtype=np.float32)
        agent_lcf_feat = np.zeros((num_agents, 9), dtype=np.float32)
        agent_fut_yaw = np.zeros((num_agents, self.future_frames), dtype=np.float32)
        
        if num_agents > 0:
            agent_lcf_feat[:, :2] = gt_boxes[:, :2]  # x, y
            agent_lcf_feat[:, 2] = gt_boxes[:, 6]  # yaw
            agent_lcf_feat[:, 3:5] = gt_boxes[:, 7:9]  # vx, vy
            agent_lcf_feat[:, 5:8] = gt_boxes[:, 3:6]  # w, l, h
            agent_lcf_feat[:, 8] = gt_labels_3d  # type
        
        gt_fut_goal = np.zeros(num_agents, dtype=np.int64)
        attr_labels = np.concatenate([
            agent_fut_trajs,
            agent_fut_masks,
            gt_fut_goal[:, None],
            agent_lcf_feat,
            agent_fut_yaw
        ], axis=-1) if num_agents > 0 else np.zeros((0, 34), dtype=np.float32)
        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names,
            attr_labels=attr_labels,
            gt_ids=gt_ids,
        )
        
        return anns_results
    
    def evaluate(self, results, metric='bbox', logger=None, jsonfile_prefix=None, 
                 result_names=['pts_bbox'], show=False, out_dir=None, pipeline=None):
        """Evaluate planning metrics (L2 and collision rate)."""
        
        print('\n')
        print('-------------- Planning Metrics --------------')
        metric_dict = None
        num_valid = 0
        
        for res in results:
            if res['metric_results']['fut_valid_flag']:
                num_valid += 1
            else:
                continue
            
            if metric_dict is None:
                metric_dict = copy.deepcopy(res['metric_results'])
            else:
                for k in res['metric_results'].keys():
                    metric_dict[k] += res['metric_results'][k]
        
        if metric_dict is not None and num_valid > 0:
            for k in metric_dict:
                if k != 'fut_valid_flag':
                    metric_dict[k] = metric_dict[k] / num_valid
                print(f"{k}: {metric_dict[k]}")
        else:
            print("No valid samples for evaluation")
        
        return metric_dict if metric_dict is not None else {}
