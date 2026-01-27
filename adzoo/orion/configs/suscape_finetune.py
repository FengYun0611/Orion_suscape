# ------------------------------------------------------------------------
# SUScape Fine-tuning Configuration for ORION with ShareGPT QA Integration
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

# Object classes for SUScape
NameMapping = {
    'Vehicle': 'car',
    'Pedestrian': 'pedestrian',
    'Bicycle': 'bicycle',
}

# Use same 9 classes as pretrained checkpoint
class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian', 'others'
]

eval_cfg = {
    "dist_ths": [0.5, 1.0, 2.0, 4.0],
    "dist_th_tp": 2.0,
    "min_recall": 0.1,
    "min_precision": 0.1,
    "mean_ap_weight": 5,
    "class_names": ['car', 'pedestrian', 'bicycle'],
    "tp_metrics": ['trans_err', 'scale_err', 'orient_err', 'vel_err'],
    "err_name_maping": {'trans_err': 'mATE', 'scale_err': 'mASE', 'orient_err': 'mAOE', 'vel_err': 'mAVE', 'attr_err': 'mAAE'},
    "class_range": {'car': (50, 50), 'pedestrian': (40, 40), 'bicycle': (40, 40)}
}

queue_length = 1
use_memory = True
num_gpus = 4  # Adjust based on your GPU availability
batch_size = 2  # Per GPU batch size
num_iters_per_epoch = 1000  # Adjust based on dataset size
num_epochs = 10
llm_path = './ckpts/pretrain_qformer/'
use_gen_token = True
use_col_loss = True
collect_keys = ['lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'command']
mix_qa_training = True  # Enable QA training mode

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

model = dict(
    type='Orion',
    save_path='./results_suscape_finetune/',
    use_grid_mask=True,
    frozen=False,
    use_lora=True,
    tokenizer=llm_path,
    lm_head=llm_path,
    use_gen_token=use_gen_token,
    use_diff_decoder=False,
    use_col_loss=use_col_loss,
    loss_plan_reg=dict(type='L1Loss', loss_weight=3.0),
    loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=3.0, dis_thresh=1.0),
    loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0),
    loss_vae_gen=dict(type='ProbabilisticLoss', loss_weight=3.0),
    mix_qa_training=mix_qa_training,
    use_critical_qa=True,  # Use QA as LLM context during training
    qa_pretrain=False,  # Fine-tuning mode (not pretraining)
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
            flash_attn=True,),
        train_cfg=dict(
            assigner=dict(
                type='LaneHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.5),
                reg_cost=dict(type='LaneL1Cost', weight=0.02),
                iou_cost=dict(type='IoUCost', weight=0.0))),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.02),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.0)),
    pts_bbox_head=dict(
        type='OrionHead',
        num_classes=9,
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
            classes=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            compensate=[0, 0.3, 0.3, 0, 0, 0, 0, 0.3, 0],
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
            num_classes=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_traffic=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range),)
    )
)

dataset_type = "SUScapeOrionDataset"

# Data paths
data_root = "/lab/haoq_lab/cse12311753/suscape_scenes/"
csv_root = "/lab/haoq_lab/cse12311753/suscape_scene_traj_csv_alldistance_fixyaw/"
qa_root = "/lab/haoq_lab/cse12311753/sharegpt_dataset/"
qa_tasks = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"]

# Use the existing pkl file from the data directory
# If you have suscape_infos_test.pkl, use that. Otherwise use test.pkl
info_root = "data/suscape_infos"

file_client_args = dict(backend="disk")
# Use existing suscape_infos_test.pkl for both train and val
# This will be split into train/val by split_suscape_data.py
ann_file_train = info_root + "/suscape_infos_train.pkl"  # Will be created by split script
ann_file_val = info_root + "/suscape_infos_val.pkl"      # Will be created by split script

# Training pipeline with QA integration
train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),
    # Load QA conversations for training
    dict(type='LoadAnnoatationCriticalVQATest',  # Loads QA as LLM context
         load_type=["critical_qa"],
         tokenizer=llm_path,
         use_gen_token=use_gen_token,
         max_length=2048,),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PETRFormatBundle3D", class_names=class_names, collect_keys=collect_keys),
    dict(type='CustomCollect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs', 'input_ids',
               'gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
               'ego_lcf_feat', 'vlm_labels', 'can_bus'] + collect_keys),
]

# Validation pipeline
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
    train=dict(
        type=dataset_type,
        seq_mode=False,  # No temporal sequence for SUScape
        data_root=data_root,
        csv_root=csv_root,
        qa_root=qa_root,
        qa_tasks=qa_tasks,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        name_mapping=NameMapping,
        modality=input_modality,
        queue_length=queue_length,
        past_frames=past_frames,
        future_frames=future_frames,
        point_cloud_range=point_cloud_range,
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        box_type_3d='LiDAR',
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        csv_root=csv_root,
        qa_root=qa_root,
        qa_tasks=qa_tasks,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        classes=class_names,
        name_mapping=NameMapping,
        modality=input_modality,
        queue_length=queue_length,
        past_frames=past_frames,
        future_frames=future_frames,
        point_cloud_range=point_cloud_range,
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        eval_cfg=eval_cfg,
        test_mode=True,
    ),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

# Optimizer with lower learning rate for fine-tuning
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=1e-5,  # 10x lower than pretraining (was 8e-5)
    betas=(0.9, 0.999),
    weight_decay=1e-5,
    paramwise_cfg={
        'decay_rate': 0.9,
        'head_decay_rate': 4.0,
        'lm_head_decay_rate': 0.1,
        'decay_type': 'vit_wise',
        'num_layers': 24,
    })

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

evaluation = dict(interval=num_iters_per_epoch, pipeline=test_pipeline)
find_unused_parameters = False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)

runner = dict(
    type='IterBasedRunner',
    max_iters=num_epochs * num_iters_per_epoch)

log_config = dict(
    interval=10,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

# Start from pretrained ORION Stage 3 checkpoint
load_from = 'ckpts/orion_stage3.pth'
resume_from = None
