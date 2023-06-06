cvpr_data_path = "/home/user/cvpr_porject/raw_data"
dataset_name = "DATASET_NAME"

classes_dict = dict( Cable      = ('break', 'thunderbolt'), 
                     Capacitor  = ('0',), 
                     Casting    = ('Inclusoes', 'Rechupe'), 
                     Console    = ('Collision', 'Dirty', 'Gap', 'Scratch'), 
                     Cylinder   = ('Chip', 'PistonMiss', 'Porosity', 'RCS'), 
                     Electronics= ('damage',), 
                     Groove     = ('s_burr', 's_scratch'), 
                     Hemisphere = ('Defect-A', 'Defect-B', 'Defect-C', 'Defect-D'), 
                     Lens       = ('Fiber', 'Flash Particle', 'Hole', 'Surface Damage', 'Tear'), 
                     PCB_1      = ('missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper'), 
                     PCB_2      = ('defect1', 'defect2', 'defect3', 'defect4', 'defect5', 'defect6', 'defect7'), 
                     Ring       = ('t_contamination', 't_scratch', 'unfinished_surface'), 
                     Screw      = ('defect',), 
                     Wood       = ('impurities', 'pits'))

train_resize = [(1024, 1024), (1536,1536),(2048, 2048), (2560,2560),(3072, 3072),(3584,3584),(4096,4096)]
train_crop_size = (2048,2048)
test_resize = [(1024, 1024), (1536,1536),(2048, 2048), (2560,2560),(3072, 3072),(3584,3584),(4096,4096)]

samples_per_gpu = 1
workers_per_gpu = 4
save_interval = 1
log_interval = 10
eval_interval = 6
lr_init = 1e-4
train_repeat_times = 5
max_epochs = 6
lr_step = [1,4]

train_subpath = "train_val"
val_subpath = "train_val"
test_subpath = "inference"

#########################################################################################
dataset_type = 'CocoDataset'
dataset_classes = classes_dict[dataset_name]
num_classes=len(dataset_classes)
train_path = f"{cvpr_data_path}/{dataset_name}/{train_subpath}/"
train_json = f"{cvpr_data_path}/{dataset_name}/{train_subpath}/_annotations.coco.json"
val_path = f"{cvpr_data_path}/{dataset_name}/{val_subpath}/"
val_json = f"{cvpr_data_path}/{dataset_name}/{val_subpath}/_annotations.coco.json"
test_path = f"{cvpr_data_path}/{dataset_name}/{test_subpath}/"
test_json = f"{cvpr_data_path}/{dataset_name}/{test_subpath}/_annotations.coco.json"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
###################################################################################################

model = dict(
    type='CascadeRCNN',
    pretrained=None,
    backbone=dict(
        type='CBSwinTransformer',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        frozen_stages=2),
    neck=dict(
        type='CBFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.4,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.0001),
            max_per_img=200,
            mask_thr_binary=0.5))
)



train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=train_resize,
        multiscale_mode='value',
        keep_ratio=True
    ),
    dict(type='RandomCrop', crop_type='absolute_range', crop_size=train_crop_size, allow_negative_crop=False),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical']),  #
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=test_resize,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type='RepeatDataset',
        times=train_repeat_times,
        dataset=dict(
            type=dataset_type,
            classes=dataset_classes,
            img_prefix=train_path,
            ann_file=train_json,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        classes=dataset_classes,
        img_prefix=val_path,
        ann_file=val_json,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=dataset_classes,
        img_prefix=test_path,
        ann_file=test_json,
        pipeline=test_pipeline),
    persistent_workers=True)

optimizer = dict(
    type='AdamW',
    lr=lr_init,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                    'relative_position_bias_table': dict(decay_mult=0.),
                                    'norm': dict(decay_mult=0.),
                                    # 'backbone': dict(lr_mult=0.1, decay_mult=1.0)
                                    }))
optimizer_config = dict(
    grad_clip=None,
    # type='Fp16OptimizerHook',
    # loss_scale='dynamic',
)

lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=100,
    # warmup_ratio=0.001,
    step=lr_step
)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
evaluation = dict(metric=['bbox', 'segm'], interval=eval_interval)
checkpoint_config = dict(interval=save_interval)
# yapf:disable
log_config = dict(
    interval=log_interval,
    hooks=[dict(type='TextLoggerHook')]
)
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "coco_weight/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth"
resume_from = None
workflow = [('train', 1)]
work_dir = f"result/0.weight/{dataset_name}"