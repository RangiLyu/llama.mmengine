# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=10)
val_cfg = dict(type='ValLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='LinearLR',
        begin=0,
        end=3,
        by_epoch=True,
        start_factor=1.0,
        end_factor=0.0,
        convert_to_iter_based=True
        )
]

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=3e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    )
