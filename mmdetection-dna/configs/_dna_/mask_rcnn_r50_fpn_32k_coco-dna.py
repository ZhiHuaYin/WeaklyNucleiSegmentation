_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_dna.py',
    '../_base_/schedules/schedule_32k.py',
    '../_base_/default_runtime_dna.py',
]
num_classes = 1
# model settings
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        frozen_stages=1,
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4
    ),
    rpn_head=dict(
        type='RPNHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2],
            ratios=[0.8, 1.0, 1.25],
            strides=[4, 8, 16, 32]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2)),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=num_classes,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            sampler=dict(
                type='RandomSampler',
                pos_fraction=0.75)),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000),
        rcnn=dict(
            sampler=dict(
                type='RandomSampler',
                num=768)),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=350,
            mask_thr_binary=0.5)
    )
)
evaluation = dict(metric=['bbox', 'segm'])
