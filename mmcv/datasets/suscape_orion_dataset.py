# ------------------------------------------------------------------------
# SUScape Dataset for ORION Open-Loop Evaluation
# Copyright (c) Xiaomi, Inc. All rights reserved.
# ------------------------------------------------------------------------
"""
SUScape Dataset for ORION Model Evaluation

This dataset handles the SUScape dataset format which includes:
- CSV files for vehicle trajectories
- Multi-view camera images from scene directories
- QA dataset (optional but recommended) for precise GT trajectories
- No map information available

Supported Data Structures:

1. Old structure (with raws/ subdirectory):
    data_root/
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

2. New structure (separate CSV directory):
    data_root/
    ├── scene-000000/
    │   ├── CAM_FRONT/
    │   ├── CAM_FRONT_LEFT/
    │   ├── CAM_FRONT_RIGHT/
    │   ├── CAM_BACK/
    │   ├── CAM_BACK_LEFT/
    │   └── CAM_BACK_RIGHT/
    ├── scene-000001/
    └── ...
    
    csv_root/
    ├── 0.csv      # Corresponds to scene-000000
    ├── 1.csv      # Corresponds to scene-000001
    └── ...
    
    qa_root/ (optional but recommended for better GT)
    ├── dataset_info.json
    ├── suscape_NQA_q1.json   # VRU identification
    ├── suscape_NQA_q7.json   # Trajectory prediction (used for L2 evaluation)
    └── suscape_NQA_qX.json   # Other tasks

For new structure, set csv_root parameter to the directory containing CSV files.
For QA-based evaluation, set qa_root and qa_tasks parameters.
"""

import copy
import numpy as np
import os
from os import path as osp
import torch
import pandas as pd
import pickle
import json
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
    collision rates. Optionally loads QA dataset for more precise GT.
    
    Args:
        data_root (str): Root directory containing scene folders
            Old structure: Should point to parent of 'raws/' directory
            New structure: Should point to directory containing scene-XXXXXX folders
        ann_file (str): Path to annotation file (will be generated if not exists)
        csv_root (str, optional): Directory containing CSV files (for new structure)
            If None, expects old structure with CSV in each scene directory
            If set, expects CSV files named {scene_num}.csv (e.g., 0.csv, 1.csv, ...)
        qa_root (str, optional): Path to QA dataset directory (e.g., 'sharegpt_dataset')
            Contains suscape_NQA_q*.json files for various tasks
            Highly recommended for accurate trajectory GT (q7 task)
        qa_tasks (list, optional): List of QA tasks to load (e.g., ['q7'] for trajectory)
            Available tasks: q1-q12 (see documentation)
            q7 provides precise ego trajectory points for L2 evaluation
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
        csv_root=None,  # Path to directory containing CSV files (e.g., 'suscape_scene_traj_csv_alldistance_fixyaw')
        qa_root=None,  # Path to QA dataset directory (e.g., 'sharegpt_dataset')
        qa_tasks=None,  # List of QA tasks to load (e.g., ['q7'] for trajectory prediction)
        *args,
        eval_mode=['det'],
        **kwargs
    ):
        # Set attributes BEFORE calling super().__init__() because parent's __init__
        # calls load_annotations() which needs these attributes
        self.csv_root = csv_root  # Store CSV root directory
        self.qa_root = qa_root  # Store QA dataset root directory
        self.qa_tasks = qa_tasks if qa_tasks is not None else []
        self.qa_data = {}  # Will store loaded QA data
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.point_cloud_range = np.array(point_cloud_range)
        self.polyline_points_num = polyline_points_num
        
        super().__init__(*args, **kwargs)
        
        self.queue_length = queue_length
        self.with_velocity = with_velocity
        self.NameMapping = name_mapping if name_mapping is not None else {}
        self.eval_cfg = eval_cfg if eval_cfg is not None else {}
        self.sample_interval = sample_interval
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
        
        # Load QA dataset if specified
        if self.qa_root and self.qa_tasks:
            self._load_qa_dataset()
    
    def _load_qa_dataset(self):
        """Load QA dataset JSON files for specified tasks.
        
        Supports multiple ID formats:
        - scene-000000_frame_0000 (old format)
        - scene-000000_7_q7 (new format, where 7 is frame number)
        """
        print(f"Loading QA dataset from {self.qa_root} for tasks: {self.qa_tasks}")
        
        for task in self.qa_tasks:
            qa_file = osp.join(self.qa_root, f'suscape_NQA_{task}.json')
            if osp.exists(qa_file):
                with open(qa_file, 'r') as f:
                    qa_list = json.load(f)
                    # Index QA data by scene and frame for quick lookup
                    # QA data structure: list of dicts with 'id', 'image', 'conversations', etc.
                    self.qa_data[task] = {}
                    for qa_item in qa_list:
                        # Extract scene and frame from id
                        # Support multiple formats:
                        # 1. scene-000000_frame_0000 (old format)
                        # 2. scene-000000_7_q7 (new format, where 7 is frame number)
                        item_id = qa_item.get('id', '')
                        
                        scene_name = None
                        frame_idx = None
                        
                        if '_frame_' in item_id:
                            # Old format: scene-000000_frame_0000
                            scene_name, frame_part = item_id.split('_frame_')
                            frame_idx = int(frame_part)
                        elif f'_{task}' in item_id:
                            # New format: scene-000000_7_q7
                            # Split by underscore and extract scene and frame
                            parts = item_id.split('_')
                            if len(parts) >= 3:
                                # scene-000000_7_q7 -> ['scene-000000', '7', 'q7']
                                scene_name = parts[0]
                                try:
                                    frame_idx = int(parts[1])
                                except (ValueError, IndexError):
                                    continue
                        else:
                            # Try to parse generic format
                            parts = item_id.split('_')
                            if len(parts) >= 2:
                                scene_name = parts[0]
                                try:
                                    frame_idx = int(parts[1])
                                except (ValueError, IndexError):
                                    continue
                        
                        if scene_name and frame_idx is not None:
                            if scene_name not in self.qa_data[task]:
                                self.qa_data[task][scene_name] = {}
                            self.qa_data[task][scene_name][frame_idx] = qa_item
                    
                print(f"Loaded {len(qa_list)} QA items for task {task}")
            else:
                print(f"Warning: QA file not found: {qa_file}")
    
    def _get_qa_trajectory(self, scene_name, frame_idx, task='q7'):
        """Extract trajectory from QA data for a specific scene and frame.
        
        Args:
            scene_name (str): Scene identifier (e.g., 'scene-000000')
            frame_idx (int): Frame index
            task (str): QA task ID (default: 'q7' for trajectory prediction)
            
        Returns:
            numpy array or None: Trajectory points if available, else None
        """
        if task not in self.qa_data:
            return None
        
        if scene_name not in self.qa_data[task]:
            return None
        
        if frame_idx not in self.qa_data[task][scene_name]:
            return None
        
        qa_item = self.qa_data[task][scene_name][frame_idx]
        
        # Extract trajectory from conversations
        # Format varies by task, q7 should have trajectory in structured format
        conversations = qa_item.get('conversations', [])
        for conv in conversations:
            if conv.get('from') == 'gpt':  # GT answer
                value = conv.get('value', '')
                # Parse trajectory from the answer
                # Expected format for q7: structured waypoints or coordinates
                traj = self._parse_trajectory_from_qa(value)
                if traj is not None:
                    return traj
        
        return None
    
    def _parse_trajectory_from_qa(self, qa_text):
        """Parse trajectory coordinates from QA text.
        
        The q7 task should provide trajectory in a structured format.
        This method extracts numerical coordinates.
        
        Supports multiple formats:
        - [x, y]: [5.15, 0.02], [10.31, 0.05], ... (SUScape format)
        - (x, y), (x, y), ...
        - x: value, y: value
        
        Note: SUScape coordinate system: +x = forward, +y = left (in meters)
        
        Args:
            qa_text (str): QA answer text containing trajectory
            
        Returns:
            numpy array or None: Parsed trajectory points [N, 2] or None
        """
        import re
        
        # Primary pattern: [x, y] format (SUScape format)
        # Matches: [5.15, 0.02] or [10.31, 0.05]
        coord_pattern = r'\[(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]'
        matches = re.findall(coord_pattern, qa_text)
        
        if matches:
            trajectory = np.array([[float(x), float(y)] for x, y in matches], dtype=np.float32)
            return trajectory
        
        # Fallback pattern: (x, y) format
        coord_pattern = r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'
        matches = re.findall(coord_pattern, qa_text)
        
        if matches:
            trajectory = np.array([[float(x), float(y)] for x, y in matches], dtype=np.float32)
            return trajectory
        
        # Try alternative format: "x: value, y: value"
        xy_pattern = r'x\s*:\s*(-?\d+\.?\d*)\s*,\s*y\s*:\s*(-?\d+\.?\d*)'
        matches = re.findall(xy_pattern, qa_text, re.IGNORECASE)
        
        if matches:
            trajectory = np.array([[float(x), float(y)] for x, y in matches], dtype=np.float32)
            return trajectory
        
        return None

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
        """Generate annotations by scanning SUScape scene directories and CSV files.
        
        Supports two directory structures:
        1. Old structure: data_root/raws/scene-XXXXXX/0.csv + cameras
        2. New structure: data_root/scene-XXXXXX/cameras + csv_root/X.csv
        """
        # Try new structure first (scenes in data_root, CSVs in csv_root)
        if self.csv_root is not None:
            return self._generate_annotations_new_structure()
        
        # Fall back to old structure (raws/ subdirectory)
        raws_dir = osp.join(self.data_root, 'raws')
        if osp.exists(raws_dir):
            return self._generate_annotations_old_structure(raws_dir)
        
        # Try scenes directly in data_root
        scene_dirs = sorted([d for d in os.listdir(self.data_root) 
                           if osp.isdir(osp.join(self.data_root, d)) and d.startswith('scene-')])
        if scene_dirs:
            print(f'Warning: Found scenes in {self.data_root} but no csv_root specified.')
            print(f'Please set csv_root parameter to the directory containing CSV files.')
            raise ValueError(f'csv_root parameter is required when scenes are in {self.data_root}')
        
        raise FileNotFoundError(f'No SUScape scenes found in {self.data_root} or {raws_dir}')
    
    def _generate_annotations_old_structure(self, raws_dir):
        """Generate annotations from old structure: raws/scene-XXXXXX/0.csv + cameras"""
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
    
    def _generate_annotations_new_structure(self):
        """Generate annotations from new structure: data_root/scene-XXXXXX/cameras + csv_root/X.csv
        
        In this structure:
        - Scenes are directly in data_root (e.g., data_root/scene-000000/)
        - CSV files are in csv_root with numeric names (e.g., csv_root/0.csv, 1.csv, ...)
        - CSV filename number corresponds to scene number (0.csv -> scene-000000)
        """
        data_infos = []
        
        # Find all scene directories
        scene_dirs = sorted([d for d in os.listdir(self.data_root) 
                           if osp.isdir(osp.join(self.data_root, d)) and d.startswith('scene-')])
        
        print(f'Found {len(scene_dirs)} scenes in {self.data_root}')
        print(f'Looking for CSV files in {self.csv_root}')
        
        for scene_name in mmcv.track_iter_progress(scene_dirs):
            # Extract scene number from scene name (e.g., 'scene-000000' -> 0)
            try:
                scene_num = int(scene_name.split('-')[1])
            except (IndexError, ValueError):
                print(f'Warning: Cannot parse scene number from {scene_name}, skipping...')
                continue
            
            # Look for corresponding CSV file (e.g., 0.csv for scene-000000)
            csv_file = osp.join(self.csv_root, f'{scene_num}.csv')
            if not osp.exists(csv_file):
                print(f'Warning: CSV file not found: {csv_file}, skipping...')
                continue
            
            scene_dir = osp.join(self.data_root, scene_name)
            
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
        
        # Debug: Print column names if TIMESTAMP is missing
        if 'TIMESTAMP' not in df.columns:
            print(f"ERROR: CSV file {csv_file} columns: {df.columns.tolist()}")
            print(f"First few rows:\n{df.head()}")
            raise KeyError(f"'TIMESTAMP' column not found in CSV file {csv_file}. Available columns: {df.columns.tolist()}")
        
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
        """Build frame information dictionary.
        
        Supports multiple camera directory structures:
        1. Old: scene-XXXXXX/CAM_FRONT/
        2. New: scene-XXXXXX/camera/front/
        
        Image naming:
        - Old: Frame index-based (0.jpg, 1.jpg, ...)
        - New: Timestamp-based (1630376940.500.jpg, ...)
        """
        
        # Ego vehicle information
        ego_x, ego_y = ego_data['X'], ego_data['Y']
        ego_vx, ego_vy = ego_data['V_X'], ego_data['V_Y']
        ego_ax, ego_ay = ego_data['A_X'], ego_data['A_Y']
        ego_yaw = ego_data['YAW'] * np.pi / 180  # Convert to radians
        ego_dyaw = ego_data['DYAW'] * np.pi / 180
        
        # Build camera paths (assuming standard camera naming)
        # Support both old (CAM_FRONT) and new (camera/front) structures
        camera_mapping = {
            'CAM_FRONT': 'front',
            'CAM_FRONT_LEFT': 'front_left',
            'CAM_FRONT_RIGHT': 'front_right',
            'CAM_BACK': 'rear',
            'CAM_BACK_LEFT': 'rear_left',
            'CAM_BACK_RIGHT': 'rear_right',
        }
        
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
        for cam_name, cam_subdir in camera_mapping.items():
            # Try old structure first: scene-XXXXXX/CAM_FRONT/
            cam_dir = osp.join(scene_dir, cam_name)
            img_path = None
            
            if osp.exists(cam_dir):
                # Old structure - use frame index for image filename
                img_files = sorted([f for f in os.listdir(cam_dir) if f.endswith(('.jpg', '.png'))])
                if frame_idx < len(img_files):
                    if 'raws' in scene_dir:
                        img_path = osp.join('raws', scene_name, cam_name, img_files[frame_idx])
                    else:
                        img_path = osp.join(scene_name, cam_name, img_files[frame_idx])
            else:
                # Try new structure: scene-XXXXXX/camera/front/
                cam_dir = osp.join(scene_dir, 'camera', cam_subdir)
                if osp.exists(cam_dir):
                    # New structure - use timestamp for image filename
                    # Image filename format: timestamp.jpg (e.g., 1630376940.500.jpg)
                    img_filename = f'{timestamp:.3f}.jpg'
                    img_file = osp.join(cam_dir, img_filename)
                    
                    # Check if file exists
                    if osp.exists(img_file):
                        if 'raws' in scene_dir:
                            img_path = osp.join('raws', scene_name, 'camera', cam_subdir, img_filename)
                        else:
                            img_path = osp.join(scene_name, 'camera', cam_subdir, img_filename)
                    else:
                        # Fallback: try to find image by index
                        img_files = sorted([f for f in os.listdir(cam_dir) if f.endswith(('.jpg', '.png'))])
                        if frame_idx < len(img_files):
                            if 'raws' in scene_dir:
                                img_path = osp.join('raws', scene_name, 'camera', cam_subdir, img_files[frame_idx])
                            else:
                                img_path = osp.join(scene_name, 'camera', cam_subdir, img_files[frame_idx])
            
            if img_path is not None:
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
        """Get ego vehicle historical and future trajectories.
        
        Priority:
        1. QA dataset (q7) if available - provides precise GT trajectories
        2. CSV data - extracted from vehicle positions
        """
        scene_token = self.data_infos[index]['folder']
        frame_idx = self.data_infos[index]['frame_idx']
        
        # Initialize trajectories
        ego_his_trajs = np.zeros((past_frames, 2), dtype=np.float32)
        ego_fut_trajs = np.zeros((future_frames, 2), dtype=np.float32)
        ego_fut_masks = np.zeros(future_frames, dtype=np.float32)
        
        current_pos = self.data_infos[index]['ego_translation'][:2]
        current_yaw = self.data_infos[index]['ego_yaw']
        
        # Try to get future trajectory from QA dataset (q7) if available
        qa_fut_traj = None
        if 'q7' in self.qa_tasks and 'q7' in self.qa_data:
            qa_fut_traj = self._get_qa_trajectory(scene_token, frame_idx, task='q7')
        
        if qa_fut_traj is not None and len(qa_fut_traj) > 0:
            # Use QA trajectory - already in ego frame or world frame
            # Assume QA trajectory is in world frame, transform to ego frame
            for i in range(min(len(qa_fut_traj), future_frames)):
                # QA trajectory might already be relative or in world coords
                # Try to use it directly first, assuming it's in ego frame
                if qa_fut_traj[i][0] < 100 and qa_fut_traj[i][1] < 100:  # Sanity check for ego frame
                    ego_fut_trajs[i] = qa_fut_traj[i][:2]
                    ego_fut_masks[i] = 1.0
                else:
                    # Transform from world to ego frame
                    delta = qa_fut_traj[i][:2] - current_pos
                    cos_yaw, sin_yaw = np.cos(-current_yaw), np.sin(-current_yaw)
                    ego_fut_trajs[i] = [
                        cos_yaw * delta[0] - sin_yaw * delta[1],
                        sin_yaw * delta[0] + cos_yaw * delta[1]
                    ]
                    ego_fut_masks[i] = 1.0
            print(f"Using QA trajectory for scene {scene_token} frame {frame_idx}")
        else:
            # Fallback to CSV-based trajectory extraction
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
        
        # Get historical trajectory (always from CSV)
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
        """Evaluate planning metrics (L2 and collision rate).
        
        Computes metrics at 1s, 2s, 3s and their averages.
        """
        
        print('\n')
        print('=' * 60)
        print('Planning Metrics Evaluation (SUScape QA Dataset)')
        print('=' * 60)
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
            # Compute averages
            for k in metric_dict:
                if k != 'fut_valid_flag':
                    metric_dict[k] = metric_dict[k] / num_valid
            
            # Compute average across time horizons
            metric_dict['plan_L2_avg'] = (
                metric_dict['plan_L2_1s'] + 
                metric_dict['plan_L2_2s'] + 
                metric_dict['plan_L2_3s']
            ) / 3.0
            
            metric_dict['plan_obj_col_avg'] = (
                metric_dict['plan_obj_col_1s'] + 
                metric_dict['plan_obj_col_2s'] + 
                metric_dict['plan_obj_col_3s']
            ) / 3.0
            
            metric_dict['plan_obj_box_col_avg'] = (
                metric_dict['plan_obj_box_col_1s'] + 
                metric_dict['plan_obj_box_col_2s'] + 
                metric_dict['plan_obj_box_col_3s']
            ) / 3.0
            
            # Print results in organized format
            print(f"\nTotal valid samples: {num_valid}\n")
            
            print("L2 Trajectory Error (meters):")
            print(f"  1s: {metric_dict['plan_L2_1s']:.4f}")
            print(f"  2s: {metric_dict['plan_L2_2s']:.4f}")
            print(f"  3s: {metric_dict['plan_L2_3s']:.4f}")
            print(f"  Avg: {metric_dict['plan_L2_avg']:.4f}")
            
            print("\nObject Collision Rate:")
            print(f"  1s: {metric_dict['plan_obj_col_1s']:.4f}")
            print(f"  2s: {metric_dict['plan_obj_col_2s']:.4f}")
            print(f"  3s: {metric_dict['plan_obj_col_3s']:.4f}")
            print(f"  Avg: {metric_dict['plan_obj_col_avg']:.4f}")
            
            print("\nBounding Box Collision Rate:")
            print(f"  1s: {metric_dict['plan_obj_box_col_1s']:.4f}")
            print(f"  2s: {metric_dict['plan_obj_box_col_2s']:.4f}")
            print(f"  3s: {metric_dict['plan_obj_box_col_3s']:.4f}")
            print(f"  Avg: {metric_dict['plan_obj_box_col_avg']:.4f}")
            
            print('\n' + '=' * 60)
        else:
            print("No valid samples for evaluation")
            metric_dict = {}
        
        return metric_dict
