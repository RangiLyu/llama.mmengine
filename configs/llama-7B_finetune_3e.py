_base_ = [
    './_base_/schedules/finetune-3e.py', './_base_/default_runtime.py'
]

model = dict(type='LoRAModel',
             model=dict(type='LLaMA7B'),
             r=8,
             alpha=16,
             dropout=0.05)

data_path = 'yahma/alpaca-cleaned'
val_set_size = 2000
tokenizer_path = 'checkpoints/mm-llama/tokenizer.model'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='seq2seq_collate'))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='seq2seq_collate'))

val_evaluator = dict(type='DummyMetric')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=3e-4),
    accumulative_counts=128//4  # TODO minibatch=4
    )
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = 4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=4)
