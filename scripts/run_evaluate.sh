########################Base Model###################
# CUDA_VISIBLE_DEVICES=0 python run_eval.py \
# --model /path-to/llama-2-7b \
# --max_memory "24GiB" \
# --eval_ppl \
# --eval_mmlu \
# --eval_tasks "wsc273,hellaswag,piqa,arc_easy,arc_challenge,boolq,winogrande,race,copa"

########################## Template ############################
CUDA_VISIBLE_DEVICES=0 python run_eval.py \
--model /path-to/llama-2-7b \
--max_memory "24GiB" \
--eval_ppl \
--eval_tasks "wsc273,hellaswag,piqa,arc_easy,arc_challenge,boolq,winogrande,race,copa,lambada,coqa" \
--param_path  /path-to-param_path/
