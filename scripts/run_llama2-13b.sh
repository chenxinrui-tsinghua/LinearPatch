CUDA_VISIBLE_DEVICES=0 python main_distill.py \
--model /path-to/llama-2-13b \
--calib_dataset wikitext2 \
--net Llama-2 \
--total_num_prune 10 \
--training_seqlen 2048 \
--val_size 16 \
--batch_size 1 \
--epochs 1 \
--num_workers 8 \
--weight_lr 1e-4 \
--save_prune_dir ./exp_save_model/llama-2-13b-prune10 \
--cache_dir ./cache/ \
--insert_type "rotate" \
--distill_type 'train_free' \
--eval_ppl \
--eval_tasks "wsc273,hellaswag,piqa,arc_easy,arc_challenge,boolq,winogrande,race,copa"

# CUDA_VISIBLE_DEVICES=0 python main_distill.py \
# --model /path-to/llama-2-13b \
# --calib_dataset wikitext2 \
# --net Llama-2 \
# --total_num_prune 10 \
# --train_size 5000 \
# --training_seqlen 2048 \
# --val_size 16 \
# --batch_size 1 \
# --epochs 1 \
# --num_workers 8 \
# --weight_lr 1e-4 \
# --save_prune_dir ./exp_save_model/llama-2-13b-prune10 \
# --cache_dir ./cache/ \
# --insert_type "rotate" \
# --distill_type 'output_kl' \
# --eval_ppl \
# --eval_tasks "wsc273,hellaswag,piqa,arc_easy,arc_challenge,boolq,winogrande,race,copa"