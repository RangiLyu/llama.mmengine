_base_ = [
    './_base_/schedules/finetune-5e.py', './_base_/default_runtime.py'
]

model = dict(type='LLaMA-toy')

data_path = 'yahma/alpaca-cleaned'
val_set_size = 2000
tokenizer_path = 'decapoda-research/llama-7b-hf'

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
    optimizer=dict(type='AdamW', lr=1e-7),
    accumulative_counts=128//4  # TODO minibatch=4
    )