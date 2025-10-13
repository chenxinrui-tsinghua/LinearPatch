import os
import sys
import random
import numpy as np
import torch
import time
from datautils_block import get_loaders, test_ppl, get_offline_dataset
import torch.nn as nn
from efficient_distill import distill_loop
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
# from accelerate import infer_auto_device_map, dispatch_model
from layer_select import select_layer
from block_scaling import register_linear_patch
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.backends.cudnn.benchmark = True


def save_model(logger, tokenizer, pmodel, args):
    logger.info("start saving model")
    tokenizer.save_pretrained(args.save_prune_dir)
    pmodel.save_pretrained(args.save_prune_dir)
    config = AutoConfig.from_pretrained(args.save_prune_dir, trust_remote_code=True)
    config.num_hidden_layers = config.num_hidden_layers - args.total_num_prune
    config._name_or_path = args.save_prune_dir
    config.save_pretrained(args.save_prune_dir)
    logger.info("save model success")

@torch.no_grad()
def evaluate(model, tokenizer, args, logger):
    '''
    Note: evaluation simply move model to single GPU.
    '''
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.to(dev)

    # block_class_name = model.model.layers[0].__class__.__name__
    # device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    # model = dispatch_model(model, device_map=device_map)
    results = {}

    if args.eval_ppl:
        datasets = ["wikitext2", "c4","ptb"]
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
            print(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.eval_batch_size)
        # task_manager = lm_eval.tasks.TaskManager(
        #     include_path="/.../lm_eval_configs/tasks/",
        #     include_defaults=False) # optional if specific configs used

        results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=0,
        # task_manager=task_manager, # optional if specific configs used
        )
        res_tab = make_table(results)
        logger.info(res_tab)
        print(res_tab)
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')

    model.config.use_cache = use_cache

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument('--insert_type', type=str, choices=['diag', 'rotate', 'none'], help='insert type')
    parser.add_argument('--distill_type', type=str, choices=['output_kl', 'train_free'], help='insert type')
    parser.add_argument("--cache_dir", default="./cache", type=str, help="direction for offline distillation dataset")
    parser.add_argument("--save_prune_dir", default=None, type=str, help="direction for saving pruned model")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix", "redpajama","alpaca"],
        help="Where to extract calibration data from.")
    parser.add_argument("--train_size", type=int, default=5000, help="Number of training data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--total_num_prune", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=1,help="multi porcess to load data")
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="input sequence length for evaluating perplexity")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2, ptb and c4")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--weight_lr", type=float, default=1e-5, help="lr of full-precision weights")
    parser.add_argument("--min_lr_factor", type=float, default=20, help="min_lr = lr/min_lr_factor")
    parser.add_argument("--wd", type=float, default=1e-4,help="weight decay")
    parser.add_argument("--net", type=str, default=None,help="model (family) name, for the easier saving of data cache")
    parser.add_argument("--max_memory", type=str, default="24GiB",help="The maximum memory of each GPU")
    parser.add_argument("--early_stop", type=int, default=0,help="early stoping after validation loss do not decrease")


    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

        
    # init logger
    model_name = args.model.split("/")[-1]
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_prune_dir:
        args.save_prune_dir = os.path.join(args.save_prune_dir,
                     f"{model_name}_prune{args.total_num_prune}_{args.net}_{args.insert_type}_{args.distill_type}_trainsize{args.train_size}")
        Path(args.save_prune_dir).mkdir(parents=True, exist_ok=True)
    logger = utils.create_logger(args.save_prune_dir, dist_rank=f"{model_name}")
    logger.info(args)
    
    if args.net is None:
        args.net = args.model.split('/')[-1]
        logger.info(f"net is None, setting as {args.net}")

        config = AutoConfig.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False)
        for param in model.parameters():
            param.requires_grad = False

        tick = time.time()
        # load calibration dataset
        cache_trainloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
        cache_valloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'
        if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
            trainloader = torch.load(cache_trainloader)
            logger.info(f"load trainloader from {cache_trainloader}")
            valloader = torch.load(cache_valloader)
            logger.info(f"load valloader from {cache_valloader}")
        else:
            trainloader, valloader = get_loaders(
                args.calib_dataset,
                tokenizer,
                args.train_size,
                args.val_size,
                seed=args.seed,
                seqlen=args.training_seqlen,
            )
            torch.save(trainloader, cache_trainloader)
            torch.save(valloader, cache_valloader)
              
        init_num_layer = len(model.model.layers)

        before_pruning_parameters = sum(p.numel() for p in model.parameters())
        num_to_prune = args.total_num_prune

        layer, start_l, end_l, similarity, scale_params, t1, t2 = select_layer(model, trainloader, num_to_prune, args.insert_type, dev)
        logger.info(f"Layer select time cost {t1}s, scale calculation time cost {t2} s")
        logger.info(f"Pruning {layer}/{init_num_layer} layers from {start_l} to {end_l} layer with similarity {similarity}")


        if args.distill_type != 'train_free' and args.insert_type != 'none':
            model_name = args.model.split("/")[-1]
            train_offline_path = f'{args.cache_dir}/trainloader_{model_name}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}/'
            train_offline_loader = get_offline_dataset(model, trainloader, end_l, train_offline_path, args, dev)
            val_offline_path = f'{args.cache_dir}/valloader_{model_name}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}/'
            val_offline_loader = get_offline_dataset(model, valloader, end_l, val_offline_path, args, dev)

        model.model.layers = nn.ModuleList([layer for i, layer in enumerate(model.model.layers) if i < start_l or i >= end_l])
        pmodel = model

        if args.insert_type != 'none':
            if args.insert_type == "rotate":
                handles = register_linear_patch(pmodel, start_l, scale_params, dev, rotated=True)
            elif args.insert_type == "diag":
                handles = register_linear_patch(pmodel, start_l, scale_params, dev, rotated=False)
            else:
                print("Invalid insert type, set to none")


        after_pruning_parameters = sum(p.numel() for p in pmodel.parameters())



        logger.info(
            "#PruneLayer: {} #Param before: {}, #Param after: {}, PruneRatio = {:.4f}%".format(end_l - start_l,
                                                                                               before_pruning_parameters,
                                                                                               after_pruning_parameters,
                                                                                               100 - 100.0 * after_pruning_parameters / before_pruning_parameters))



        if args.distill_type != 'train_free' and args.insert_type != 'none':

            pmodel = distill_loop(
                source_layer=end_l,
                pruned_model=pmodel,
                target_layer=start_l,
                args=args,
                trainloader=train_offline_loader,
                valloader=val_offline_loader,
                logger=logger,
            )
        logger.info(f"time cost {(time.time()-tick)/60} minutes")

    torch.cuda.empty_cache()
    pmodel = pmodel.cpu().half()


    if args.save_prune_dir and args.insert_type != 'none':
        for i, block in enumerate(pmodel.model.layers):
            if i == start_l - 1 and hasattr(block, 'linear_patch'):
                symmetric_weight = block.linear_patch.get_weight()
                save_param = {"scale_params":scale_params, "symmetric_weight":symmetric_weight,
                              "start_l":start_l, "end_l":end_l, "rotate": args.insert_type=="rotate"}
                save_pt_path = os.path.join(args.save_prune_dir, 'scales.pt')
                torch.save(save_param, save_pt_path)
                break


    evaluate(pmodel, tokenizer, args, logger)

    from eval.mmlu_eval import run_mmlu_eval
    mmlu_num_few_shots = [5]
    for num_few_shots in mmlu_num_few_shots:
        save_dir = os.path.join(args.save_prune_dir, f"mmlu-{num_few_shots}-shot")
        print(save_dir)
        run_mmlu_eval(model, tokenizer, model_name,
                      num_few_shots, "/path-to/datasets/mmlu_no_train/data/", save_dir)


if __name__ == "__main__":
    print(sys.argv)
    main()