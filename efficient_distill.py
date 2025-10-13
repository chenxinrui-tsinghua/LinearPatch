import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import utils
import pdb
import gc
import time
from tqdm import tqdm
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
                    
def distill_loop(
    source_layer,
    pruned_model,
    target_layer,
    args,
    trainloader,
    valloader,
    logger=None,
):
    logger.info("Starting ...")

    print("len(trainloader)={},len(valloader)={}".format(len(trainloader),len(valloader)))
    
    use_cache = pruned_model.config.use_cache
    pruned_model.config.use_cache = False


    with torch.no_grad():
        pruned_model = pruned_model.to(dev)

    loss_kl = torch.nn.KLDivLoss(reduction='batchmean')

    # start distillation
    logger.info(f"=== Start distill Student block {target_layer} with Teacher blocks {source_layer}===")
    total_training_iteration = args.epochs * args.train_size / args.batch_size
    
    if args.epochs > 0:

        param_list = []

        for i, block in enumerate(pruned_model.model.layers):
            if i == target_layer - 1:
                if args.insert_type == "rotate" or args.insert_type == "diag":
                    block.linear_patch.weight.requires_grad = True
                    param_list.append(block.linear_patch.weight)


        # create optimizer
        empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
        weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration, eta_min=args.weight_lr/args.min_lr_factor)
        
        optimizer = torch.optim.AdamW(param_list, lr=args.weight_lr, weight_decay=args.wd)

        loss_scaler = utils.NativeScalerWithGradNormCount()

        best_val_loss = 1e6
        early_stop_flag = 0
        for epoch in range(args.epochs):

            loss_list = []
            norm_list = []
            start_time = time.time()
            train_loop = tqdm(enumerate(trainloader), total =len(trainloader), desc="train")
            for i, (input, target, topk_index) in train_loop:
                input, target, topk_index = input.to(dev), target.to(dev), topk_index.to(dev)
                with torch.cuda.amp.autocast():
                    s_output = pruned_model(input[0])[0]

                    s_topk_logits = torch.gather(s_output, -1, topk_index)
                    s_log_probs = torch.log_softmax(s_topk_logits, dim=-1)

                    reconstruction_loss = loss_kl(s_log_probs, target)/s_log_probs.shape[1]

                    loss =  reconstruction_loss
                    train_loop.set_postfix(loss=reconstruction_loss.detach().item())

                if not math.isfinite(loss.item()):
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()
                loss_list.append(reconstruction_loss.detach().cpu())
                optimizer.zero_grad()

                norm = loss_scaler(loss, optimizer, parameters=param_list).cpu()
                norm_list.append(norm.data)


                if args.weight_lr >0 :
                    weight_scheduler.step()
                    optimizer.param_groups[0]['lr'] = weight_scheduler.get_lr()[0]

            val_loss_list = []
            val_loop = tqdm(enumerate(valloader), total=len(valloader), desc="val")
            for i, (input, target, topk_index) in val_loop:
                input, target, topk_index = input.to(dev), target.to(dev), topk_index.to(dev)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        s_output = pruned_model(input[0])[0]
                        s_topk_logits = torch.gather(s_output, -1, topk_index)
                        s_log_probs = torch.log_softmax(s_topk_logits, dim=-1)
                        reconstruction_loss = loss_kl(s_log_probs, target) / s_log_probs.shape[1]
                    val_loss_list.append(reconstruction_loss.cpu())
            
            train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
            loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
            val_loss_mean = torch.stack(val_loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            logger.info(f"blocks {target_layer} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} lr:{weight_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
            else:
                early_stop_flag += 1
                if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                    break
        optimizer.zero_grad()
        del optimizer


    torch.cuda.empty_cache()
    gc.collect()
    pruned_model.config.use_cache = use_cache

    return pruned_model
