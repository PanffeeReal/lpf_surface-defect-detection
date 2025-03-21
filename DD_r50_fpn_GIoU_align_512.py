model = dict(
    type='FSAF',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='NASFCOS_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        norm_cfg=dict(type='BN'),
        conv_cfg=dict(type='DCNv2', deform_groups=2)),
    bbox_head=dict(
        type='DD_simple_SS_Align',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=1,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(type='TBLRBBoxCoder', normalizer=4.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction='none'),
        loss_bbox=dict(
            type='GIoULoss', eps=1e-06, loss_weight=1.0, reduction='none'),
        alpha=8,
        level_select=False,
        reg_decoded_bbox=True),
    train_cfg=dict(
        assigner=dict(
            type='CenterRegionAssigner',
            pos_scale=0.2,
            neg_scale=0.2,
            min_pos_iof=0.01),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)
dataset_type = 'GC10_DETdataset'
data_root = 'data/GC10-DET/New_GC-DET/'
classes = ('1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban', '5_youban',
           '6_siban', '7_yiwu', '8_yahen', '9_zhehen', '10_yaozhe')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type='RepeatDataset',
        classes=('1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban',
                 '5_youban', '6_siban', '7_yiwu', '8_yahen', '9_zhehen',
                 '10_yaozhe'),
        times=3,
        dataset=dict(
            type='GC10_DETdataset',
            ann_file=[
                'data/GC10-DET/New_GC-DET/VOC2012/ImageSets/trainval.txt'
            ],
            img_prefix=['data/GC10-DET/New_GC-DET/VOC2012/'],
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ])),
    val=dict(
        type='GC10_DETdataset',
        classes=('1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban',
                 '5_youban', '6_siban', '7_yiwu', '8_yahen', '9_zhehen',
                 '10_yaozhe'),
        ann_file='data/GC10-DET/New_GC-DET/VOC2012/ImageSets/test.txt',
        img_prefix='data/GC10-DET/New_GC-DET/VOC2012/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='GC10_DETdataset',
        classes=('1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban',
                 '5_youban', '6_siban', '7_yiwu', '8_yahen', '9_zhehen',
                 '10_yaozhe'),
        ann_file='data/GC10-DET/New_GC-DET/VOC2012/ImageSets/test.txt',
        img_prefix='data/GC10-DET/New_GC-DET/VOC2012/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='mAP')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7, 9, 11])
total_epochs = 12
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=12)
work_dir = './work_dirs/GC10_512_DD_GIOU_bs_16_lr_0.01_weightdecay_5e-4_momentum_0.9_epoch_8-10-12_align'
gpu_ids = range(0, 1)
