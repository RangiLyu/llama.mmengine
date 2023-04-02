import argparse
import os
import os.path as osp

from datasets import load_dataset
from mmengine.config import Config, DictAction
from transformers import DataCollatorForSeq2Seq, LlamaTokenizer

from mmllama.datasets import Prompter, seq2seq_collate
from mmllama.engine import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
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
    parser.add_argument('--local_rank', type=int, default=0)
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


    if cfg.data_path.endswith('.json') or cfg.data_path.endswith('.jsonl'):
        data = load_dataset('json', data_files=cfg.data_path)
    else:
        data = load_dataset(cfg.data_path)

    tokenizer = LlamaTokenizer(cfg.tokenizer_path)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = 'left'  # Allow batched inference

    # TODO: move hyps to cfg
    cutoff_len: int = 256
    prompt_template_name: str = 'alpaca'
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    prompter = Prompter(prompt_template_name)



    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result['input_ids'][-1] != tokenizer.eos_token_id
            and len(result['input_ids']) < cutoff_len
            and add_eos_token
        ):
            result['input_ids'].append(tokenizer.eos_token_id)
            result['attention_mask'].append(1)

        result['labels'] = result['input_ids'].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point['instruction'],
            data_point['input'],
            data_point['output'],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point['instruction'], data_point['input']
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt['input_ids'])

            tokenized_full_prompt['labels'] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt['labels'][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    train_val = data['train'].train_test_split(
        test_size=cfg.val_set_size, shuffle=True, seed=42
    )
    train_data = train_val['train'].shuffle().map(generate_and_tokenize_prompt)

    val_data = train_val['test'].shuffle().map(generate_and_tokenize_prompt)
    # collator = DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     )
    from mmengine.registry import FUNCTIONS
    FUNCTIONS.register_module(name='seq2seq_collate', module=seq2seq_collate)

    cfg.train_dataloader.dataset = train_data.remove_columns(('instruction', 'input', 'output'))
    cfg.train_dataloader.collate_fn.tokenizer = tokenizer
    cfg.val_dataloader.dataset = val_data.remove_columns(('instruction', 'input', 'output'))
    cfg.val_dataloader.collate_fn.tokenizer = tokenizer


    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    runner = Runner.from_cfg(cfg)
    # start training
    runner.train()


if __name__ == '__main__':
    main()
