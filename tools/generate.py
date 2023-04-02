import argparse

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from transformers import LlamaTokenizer

from mmllama.datasets import Prompter
from mmllama.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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
        '--instructions',
        nargs='+',
        help='instructions to generate responses')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    print_log('Building model', logger='current')
    model = MODELS.build(cfg.model)
    model.init_weights()
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'], strict=False)
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
        """Generate a response to an instruction.
        Modified from https://github.com/tloen/alpaca-lora
        """
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

    if args.instructions is not None:
        instructions = args.instructions
    else:
        instructions = [
        'Tell me about alpacas.',
        'Tell me about the president of Mexico in 2019.',
        'Tell me about the king of France in 2019.',
        'List all Canadian provinces in alphabetical order.',
        'Write a Python program that prints the first 10 Fibonacci numbers.',
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        'Count up from 1 to 500.',
    ]
    for instruction in instructions:
        print('Instruction:', instruction)
        print('Response:', evaluate(instruction))
        print()



if __name__ == '__main__':
    main()
