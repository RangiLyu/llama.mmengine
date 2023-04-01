# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)
val_cfg = dict(type='ValLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=5,
        by_epoch=True,
        milestones=[4],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-7),
    )

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = 4 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=4)
