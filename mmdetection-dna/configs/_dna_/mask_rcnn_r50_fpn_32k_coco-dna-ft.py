base_config = 'mask_rcnn_r50_fpn_32k_coco-dna'
teacher_config = 'mask_rcnn_r50_fpn_32k_coco-dna'
_base_ = [
    f'./{base_config}.py'
]
data = dict(
    train=dict(
        ann_file=f'work_dirs/{teacher_config}/iter_16000.pseudo_label.thresh0.75.iou0.00.json',
    )
)
resume_from = f'work_dirs/{teacher_config}/iter_16000.pth'
