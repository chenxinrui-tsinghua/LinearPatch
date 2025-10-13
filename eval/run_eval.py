import os
import sys
import random
import numpy as np
import torch
from datautils_block import test_ppl
import utils
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from copy import deepcopy
from block_scaling import register_linear_patch
from eval.mmlu_eval import run_mmlu_eval
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(dev)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# export HF_HOME=/dev/cache/huggingface/
# export HF_DATASETS_OFFLINE=1

@torch.no_grad()
def evaluate(model, tokenizer, logger, eval_ppl=False, eval_tasks="", eval_batch_size=1, max_memory='30GiB'):
    '''
    Note: evaluation simply move model to single GPU.
    '''
    # import pdb;pdb.set_trace()
    # block_class_name = model.model.layers[0].__class__.__name__
    # from accelerate import infer_auto_device_map, dispatch_model
    # device_map = infer_auto_device_map(model, max_memory={i: max_memory for i in range(torch.cuda.device_count())})
    # model = dispatch_model(model, device_map=device_map)
    results = {}

    if eval_ppl:
        datasets = ["wikitext2","c4","ptb"] #"c4",[] #
        ppl_results = test_ppl(model, tokenizer, datasets, 2048)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
            print(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    if eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = eval_tasks.split(',')
        model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size)

        from lm_eval import utils as lm_eval_utils
        # task_manager = lm_eval.tasks.TaskManager(include_path="/path-to/lm_eval_configs/tasks/",
        #                                          include_defaults=False)

        results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=0,
        # task_manager=task_manager
        )
        res_tab = make_table(results)
        logger.info(res_tab)
        print(res_tab)
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
    return results



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--param_path", default=None, type=str, help="scaling parameter path")
    parser.add_argument("--log_dir", default=None, type=str, help="direction of logging file")
    parser.add_argument("--eval_ppl", action="store_true", help="evaluate perplexity on wikitext2 and c4")
    parser.add_argument("--eval_tasks", type=str, default="",
                        help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--max_memory", type=str, default="70GiB", help="The maximum memory of each GPU")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map="cpu", torch_dtype=torch.float16)
    model = model.to(dev).half()
    print(model.config)

    if args.param_path is not None:
        pt_path = os.path.join(args.param_path, 'scales.pt')
        save_param = torch.load(pt_path)
        symmetric_weight = save_param["symmetric_weight"]
        start_l = save_param["start_l"]
        end_l = save_param["end_l"]
        rotate = save_param['rotate']

        handles = register_linear_patch(model, start_l, None, dev, rotated=rotate, weight=symmetric_weight)

    if args.log_dir is None:
        if args.param_path is not None:
            save_log_dir = os.path.join(args.param_path, "val_log")
        else:
            save_log_dir = os.path.join(args.model, "val_log")
    else:
        save_log_dir = args.log_dir

    model_name = args.model.split('/')[-1]
    os.makedirs(save_log_dir, exist_ok=True)
    logger = utils.create_logger(save_log_dir, dist_rank=model_name)
    print(f"log dir: {save_log_dir}")

    evaluate(model, tokenizer, logger, eval_ppl=args.eval_ppl, eval_tasks=args.eval_tasks) #,mmlu

    mmlu_num_few_shots = [5]
    for num_few_shots in mmlu_num_few_shots:
        save_dir = os.path.join(save_log_dir, f"mmlu-{num_few_shots}-shot")
        print(save_dir)
        run_mmlu_eval(model, tokenizer, model_name,
                      num_few_shots, "/path-to/datasets/mmlu_no_train/data/", save_dir)


if __name__ == "__main__":
    print(sys.argv)
    main()