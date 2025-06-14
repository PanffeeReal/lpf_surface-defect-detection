_base_ = './retinanet_r50_fpn_1x_512.py'
# model settings
model = dict(
    type='FSAF',
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
        type='FSAFHead_simple_SS_Align',
        alpha=10,
        level_select=False,
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        reg_decoded_bbox=True,
        # Only anchor-free branch is implemented. The anchor generator only
        #  generates 1 anchor at each feature point, as a substitute of the
        #  grid of features.
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=1,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(_delete_=True, type='TBLRBBoxCoder', normalizer=4.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction='none'),
        # loss_bbox=dict(
        #     _delete_=True,
        #     type='IoULoss',  ###add GIoU loss
        #     eps=1e-6,
        #     loss_weight=1.0,
        #     reduction='none')
        loss_bbox=dict(
            _delete_=True,
            type='GIoULoss',  # IoULoss
            eps=1e-6,
            loss_weight=1.0,
            reduction='none')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            # _delete_=True,
            type='CenterRegionAssigner',  ####add atss assigner to try
            pos_scale=0.2,
            neg_scale=0.2,
            min_pos_iof=0.01),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16
)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7, 9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
work_dir='./work_dirs/AEI/GC10_para_optim/alpha_10_LRSD_NASFPN_GIOU_bs_16_lr_0.01_weightdecay_5e-4_momentum_0.9_epoch_8-10-12_align'
