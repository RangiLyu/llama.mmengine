import numpy as np
from mmengine.registry import FUNCTIONS

FUNCTIONS.register_module()
def seq2seq_collate(features, tokenizer):
    # import ipdb; ipdb.set_trace()
    return_tensors = 'pt'
    label_pad_token_id: int = -100
    pad_to_multiple_of = 8
    labels = [feature['labels'] for feature in features] if 'labels' in features[0].keys() else None
    # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
    # same length to return tensors.
    if labels is not None:
        max_label_length = max(len(l) for l in labels)
        if pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )

        padding_side = tokenizer.padding_side
        for feature in features:
            remainder = [label_pad_token_id] * (max_label_length - len(feature['labels']))
            if isinstance(feature['labels'], list):
                feature['labels'] = (
                    feature['labels'] + remainder if padding_side == 'right' else remainder + feature['labels']
                )
            elif padding_side == 'right':
                feature['labels'] = np.concatenate([feature['labels'], remainder]).astype(np.int64)
            else:
                feature['labels'] = np.concatenate([remainder, feature['labels']]).astype(np.int64)
    features = tokenizer.pad(
        features,
        padding=True,
        max_length=None,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
    )

    return features
