# ------------------------------------------------------------------------
# SUScape Open-Loop Evaluation Configuration for ORION
# Copyright (c) Xiaomi, Inc. All rights reserved.
# ------------------------------------------------------------------------

_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py"]
backbone_norm_cfg = dict(type='LN', requires_grad=True)

# Point cloud range
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
   mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# SUScape does not have map classes
map_classes = []
map_fixed_ptsnum_per_gt_line = 11
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)
past_frames = 2
future_frames = 6
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2

ida_aug_conf = {
        "resize_lim": (0.37, 0.45),
        "final_dim": (320, 640),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }

# Object classes for SUScape - must match checkpoint (9 classes)
# The checkpoint was trained on nuScenes with 9 classes
# Map SUScape OBJECT_TYPE to nuScenes classes
NameMapping = {
    'Vehicle': 'car',
    'Pedestrian': 'pedestrian',
    'Bicycle': 'bicycle',
}

# Use same 9 classes as training config to match checkpoint weights
class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian', 'others'
]

eval_cfg = {
    "dist_ths": [0.5, 1.0, 2.0, 4.0],
    "dist_th_tp": 2.0,
    "min_recall": 0.1,
    "min_precision": 0.1,
    "mean_ap_weight": 5,
    # Evaluate only classes present in SUScape (car, pedestrian, bicycle)
    "class_names": ['car', 'pedestrian', 'bicycle'],
    "tp_metrics": ['trans_err', 'scale_err', 'orient_err', 'vel_err'],
    "err_name_maping": {'trans_err': 'mATE', 'scale_err': 'mASE', 'orient_err': 'mAOE', 'vel_err': 'mAVE', 'attr_err': 'mAAE'},
    "class_range": {'car': (50, 50), 'pedestrian': (40, 40), 'bicycle': (40, 40)}
}

use_memory = True
num_gpus = 1
batch_size = 1
llm_path = 'ckpts/pretrain_qformer/'
use_gen_token = True
use_col_loss = True
collect_keys = ['lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'command']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

model = dict(
    type='Orion',
    save_path='./results_suscape_eval/',
    use_grid_mask=True,
    frozen=False,
    use_lora=True,
    tokenizer=llm_path,
    lm_head=llm_path,
    use_gen_token=use_gen_token,
    use_diff_decoder=False,
    use_col_loss=use_col_loss,
    img_backbone=dict(
        type='EVAViT',
        img_size=640,
        patch_size=16,
        window_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4*2/3,
        window_block_indexes=(
            list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + 
            list(range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
        ),
        qkv_bias=True,
        drop_path_rate=0.3,
        flash_attn=True,
        with_cp=True,
        frozen=False,),
    map_head=dict(
        type='OrionHeadM',
        num_classes=0,  # No map classes for SUScape
        in_channels=1024,
        out_dims=4096,
        memory_len=600,
        with_mask=True,
        topk_proposals=300,
        num_lane=1800,
        num_lanes_one2one=300,
        k_one2many=5,
        lambda_one2many=1.0,
        num_extra=256,
        n_control=11,
        pc_range=point_cloud_range,
        code_weights=[1.0, 1.0],
        score_threshold=0.2,
        transformer=dict(
            type='PETRTemporalTransformer',
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True,)),
    pts_bbox_head=dict(
        type='OrionHead',
        num_classes=9,  # Match checkpoint: 9 classes (car, van, truck, bicycle, traffic_sign, traffic_cone, traffic_light, pedestrian, others)
        in_channels=1024,
        out_dims=4096,
        num_query=600,
        with_mask=True,
        memory_len=600,
        topk_proposals=300,
        num_propagated=300,
        num_extra=256,
        n_control=11,
        match_with_velo=False,
        pred_traffic_light_state=False,
        use_col_loss=use_col_loss,
        use_memory=use_memory,
        scalar=10,
        noise_scale=1.0,
        dn_weight=1.0,
        split=0.75,
        use_pe=False,
        motion_transformer_decoder=dict(
            type='OrionTransformerDecoder',
            num_layers=1,
            embed_dims=_dim_,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=_ffn_dim_,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False,
        ),
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        score_threshold=0.2,
        class_agnostic_nms=dict(
            classes=[0, 1, 2, 3, 4, 5, 6, 7, 8],  # All 9 classes
            compensate=[0, 0.3, 0.3, 0, 0, 0, 0, 0.3, 0],  # Compensate for van, truck, pedestrian
            pre_max_size=1000,
            post_max_size=300,
            nms_thr=0.1,
        ),
        memory_decoder_transformer=dict(
            type='OrionTransformerDecoder',
            num_layers=1,
            embed_dims=_dim_,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=_ffn_dim_,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False),
        transformer=dict(
            type='PETRTemporalTransformer',
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True,
        ),
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=9)),  # Match checkpoint: 9 classes
)

dataset_type = "SUScapeOrionDataset"

# Data paths - Update these to match your directory structure
# For new structure (separate CSV directory):
data_root = "data/suscape_scenes"  # Directory containing scene-XXXXXX folders
csv_root = "data/suscape_scene_traj_csv_alldistance_fixyaw"  # Directory containing X.csv files
qa_root = "data/sharegpt_dataset"  # QA dataset directory (optional but recommended for better GT)
qa_tasks = ["q7"]  # QA tasks to use: q7 for trajectory prediction
# For old structure (CSV in scene directories), set csv_root=None

info_root = "data/suscape_infos"
file_client_args = dict(backend="disk")
ann_file_test = info_root + f"/suscape_infos_test.pkl"

test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type='LoadAnnoatationCriticalVQATest',
         load_type=["critical_qa"],
         tokenizer=llm_path,
         use_gen_token=use_gen_token,
         max_length=2048,),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(
                type='CustomCollect3D',
                keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs', 'input_ids', 
                      'gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 
                      'ego_lcf_feat', 'vlm_labels', 'can_bus', 'fut_valid_flag'] + collect_keys,
            )]
    )
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        csv_root=csv_root,  # CSV trajectory data
        qa_root=qa_root,  # QA dataset for GT trajectories
        qa_tasks=qa_tasks,  # QA tasks to load
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        classes=class_names,
        name_mapping=NameMapping,
        modality=input_modality,
        past_frames=past_frames,
        future_frames=future_frames,
        point_cloud_range=point_cloud_range,
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        eval_cfg=eval_cfg,
    ),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
