import argparse
import os
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from transformers import LlamaTokenizer

from mmllama.datasets import Prompter
from mmllama.registry import MODELS


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    print_log('Building model', logger='current')
    model = MODELS.build(cfg.model)
    model.init_weights()
    model.load_state_dict(torch.load(args.checkpoint))
    model.half()
    model.cuda()
    model.eval()
    print_log('Finished building model', logger='current')

    tokenizer = LlamaTokenizer(cfg.tokenizer_path)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = 'left'  # Allow batched inference

    prompter = Prompter('alpaca')

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to('cuda')
        model.model.test_cfg.temperature=temperature
        model.model.test_cfg.top_k=top_k
        model.model.test_cfg.max_new_tokens=max_new_tokens
        # TODO: beam search
        with torch.no_grad():
            generation_output = model(input_ids, mode='predict')
        s = generation_output[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    # testing code for readme
    for instruction in [
        'Tell me about alpacas.',
        'Tell me about the president of Mexico in 2019.',
        'Tell me about the king of France in 2019.',
        'List all Canadian provinces in alphabetical order.',
        'Write a Python program that prints the first 10 Fibonacci numbers.',
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        'Count up from 1 to 500.',
    ]:
        print('Instruction:', instruction)
        print('Response:', evaluate(instruction))
        print()



if __name__ == '__main__':
    main()
