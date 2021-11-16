# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[20000, 27000],
    by_epoch=False
)
runner = dict(type='IterBasedRunner', max_iters=32000)
evaluation = dict(interval=32000)
