# LinearPatch: A Simple Linear Patch Revives Layer-Pruned Large Language Models

[NeurIPS 2025] Official PyTorch implementation of "A Simple Linear Patch Revives Layer-Pruned Large Language Models"


This repository contains the PyTorch implementation of the NeurIPS 2025 paper [LinearPatch: Plug-and-Play Patching for Layer-Pruned Large Language Models](https://arxiv.org/abs/2505.24680).

---

## TL;DR

LinearPatch is a **lightweight and plug-and-play technique** designed to patch **activation magnitude mismatches** in **layer-pruned LLMs**.  
By leveraging **Hadamard transforms with channel-wise scaling**, LinearPatch efficiently aligns activations across layers, achieving:  

- **Better PPL, MMLU, and QA performance** than state-of-the-art pruning methods  
- **Negligible inference overhead**  
- **Training-free or lightweight fine-tuning modes**  

<p align="center">
  <img src="figures/method_placeholder.jpg" width="600">
</p>

---

## Contents

- [Preparations](#preparations)  
- [Usage](#usage)  
- [Results](#results)  
- [References](#references)  

---

## Preparations

### Installation

```bash
conda create -n linearpatch python=3.10 -y
conda activate linearpatch
pip install -r requirements.txt
```

**Note:** - To run models like LLaMA2 and LLaMA3, please install dependencies from `requirements_llama2.txt`.

### Data Preparation

Download datasets in `./datasets`.

**Calibration set or PPL evaluation**

| Dataset  | URL                                                                                                                        |
| --------- | -------------------------------------------------------------------------------------------------------------------------- |
| WikiText2 | [https://huggingface.co/datasets/wikitext](https://huggingface.co/datasets/wikitext)                                       |
| C4        | [https://huggingface.co/datasets/allenai/c4](https://huggingface.co/datasets/allenai/c4)                                   |
| Pile      | [https://huggingface.co/datasets/mit-han-lab/pile-val-backup](https://huggingface.co/datasets/mit-han-lab/pile-val-backup) |

**Commonsense QA evaluation**

For QA evaluation, we use lm eval (0.4.4) to load datasets. if you have download the datasets, just modify the config item dataset_path in each QA dataset's config file in `~/anaconda3/envs/your-env-name/lib/python3.x/site-packages/lm_eval/tasks`.

| Dataset         | Local Dir             | URL                                                                                                              |
| --------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ARC-E and ARC-C | ./datasets/ai2_arc    | [https://huggingface.co/datasets/allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)               |
| HellaSwag       | ./datasets/hellaswag  | [https://huggingface.co/datasets/Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag)               |
| PIQA            | ./datasets/piqa       | [https://huggingface.co/datasets/ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa)                         |
| WinoGrande      | ./datasets/winogrande | [https://huggingface.co/datasets/winogrande](https://huggingface.co/datasets/winogrande)                         |
| BoolQ           | ./datasets/boolq      | [https://huggingface.co/datasets/boolq](https://huggingface.co/datasets/boolq)                                   |
| RACE            | ./datasets/race       | [https://huggingface.co/datasets/race](https://huggingface.co/datasets/race)                                     |
| COPA            | ./datasets/copa       | [https://huggingface.co/datasets/super_glue/viewer/copa](https://huggingface.co/datasets/super_glue/viewer/copa) |
| WSC             | ./datasets/wsc273     | [https://huggingface.co/datasets/wsc273](https://huggingface.co/datasets/wsc273)                                 |


### Model Preparation

Download links to officially released LLMs

| Model                          | Download Link                                                                 |
|--------------------------------|-------------------------------------------------------------------------------|
| LLaMA-2-7B                     | [https://huggingface.co/meta-llama/Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7B) |
| LLaMA-2-13B                    | [https://huggingface.co/meta-llama/Llama-2-13B](https://huggingface.co/meta-llama/Llama-2-13B) |
| LLaMA-3-8B                     | [https://huggingface.co/meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| DeepSeek-R1-Distill-Qwen-7B    | [https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| DeepSeek-R1-Distill-Llama-8B   | [https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |
| Baichuan2-7B                   | [https://huggingface.co/baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) |


## Usage

We provide example scripts for running LinearPatch.
A typical usage on LLaMA-2-7B is as follows:

```bash
bash scripts/run_llama2-7b.sh
```

The example python scripts  is as follows:

`scripts/run_llama2-7b.sh`:

```bash
CUDA_VISIBLE_DEVICES=0 python main_distill.py \
--model ./models/llama-2-7b \
--calib_dataset wikitext2 \
--net Llama-2 \
--total_num_prune 7 \
--training_seqlen 2048 \
--val_size 16 \
--batch_size 1 \
--epochs 1 \
--num_workers 8 \
--weight_lr 1e-4 \
--save_prune_dir ./exp_save_model/ \
--cache_dir ./cache/ \
--insert_type "rotate" \ # choice: ["rotate", "diag"], "diag" denotes only with diagonal matrix alignment, "rotate" denotes with Hadamard rotation, 
--distill_type "train_free" \ # choice: ["train_free", "output_kl"]
--eval_ppl \
--eval_tasks "wsc273,hellaswag,piqa,arc_easy,arc_challenge,boolq,winogrande,race,copa"
```


## Results

### PPL Results

Table 1：LLaMA-2-7B - Comparison on PPL benchmark with training-free methods (7 out of 32 layers pruned)
| Method                  | WIKI-2  | C4      | PTB     | PPL avg.  |
|-------------------------|---------|---------|---------|-----------|
| Dense                   | 5.47    | 6.97    | 22.51   | 11.65     |
| SLEB                    | 9.14    | 11.21   | 38.45   | 19.60     |
| +LinearPatch            | 8.77    | 10.66   | 38.30   | **19.24** |
| Taylor+                 | 18.45   | 20.99   | 62.18   | 33.87     |
| +LinearPatch            | 13.84   | 15.28   | 48.26   | **25.79** |
| ShotGPT                 | 18.45   | 20.99   | 62.18   | 33.87     |
| +LinearPatch            | 13.22   | 14.58   | 45.97   | **24.59** |
| LLM-Streamline(None)    | 18.45   | 20.99   | 62.18   | 33.87     |
| +LinearPatch            | 13.22   | 14.58   | 45.97   | **24.59** |



Table 2：LLaMA-3-8B - Comparison on PPL benchmark with training-free methods (7 out of 32 layers pruned)
| Method                  | WIKI-2    | C4        | PTB       | PPL avg.  |
|-------------------------|-----------|-----------|-----------|-----------|
| Dense                   | 6.14      | 8.88      | 10.59     | 8.54      |
| SLEB                    | 13.12     | 16.76     | 21.04     | 16.97     |
| +LinearPatch            | 11.97     | 15.74     | 19.55     | **15.75** |
| Taylor+                 | 2287.86   | 1491.38   | 4741.90   | 2840.38   |
| +LinearPatch            | 208.88    | 235.63    | 264.97    | **236.49**|
| ShotGPT                 | 57.76     | 50.13     | 67.39     | 58.43     |
| +LinearPatch            | 25.67     | 28.38     | 31.22     | **28.42** |
| LLM-Streamline(None)    | 2287.73   | 1491.37   | 4738.81   | 2839.30   |
| +LinearPatch            | 69.82     | 96.68     | 88.79     | **85.10** |


### QA Results

(Note: $L_p$ denotes the number of pruned layers and $L_t$ denotes the total number of layers of the model. The Ratio column represents the proportion(%) of pruning parameters to the total parameters of the model. The Avg. column denotes the average accuracy(%) and the RP column denotes the retained performance(%).)

Table 3：LLaMA-2-7B - Comparison on QA benchmark with training-free methods


| $L_p/L_t$ | Method                  | Ratio  | ARC-c  | ARC-e  | BoolQ  | HeSw   | PIQA   | WG     | WSC    | Race-h | CoPa   | Avg.       | RP         |
|-----------|-------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|------------|------------|
| 0/32      | Dense                   | -      | 46.25  | 74.58  | 77.74  | 75.97  | 79.11  | 68.98  | 80.59  | 39.62  | 87.00  | 69.98      | 100        |
| 7/32      | LLMPruner               | 20.56  | 35.24  | 60.61  | 62.42  | 61.66  | 75.41  | 54.78  | 71.43  | 31.67  | 80.00  | 59.25      | 83.80      |
| 7/32      | SLEB                    | 21.02  | 33.02  | 56.57  | 63.91  | 62.49  | 73.07  | 58.96  | 69.23  | 32.06  | 84.00  | 59.26      | 83.66      |
| 7/32      | ShortGPT                | 21.02  | 36.18  | 55.89  | 62.17  | 62.66  | 70.40  | 65.98  | 77.29  | 33.78  | 81.00  | 60.59      | 86.06      |
| 7/32      | LLM-Streamline (None)   | 21.02  | 36.18  | 55.89  | 62.17  | 62.66  | 70.40  | 65.98  | 77.29  | 33.78  | 81.00  | 60.59      | 86.06      |
| 7/32      | ShortGPT＋LinearPatch   | 20.78  | 37.63  | 61.24  | 62.14  | 63.49  | 70.46  | 65.90  | 79.49  | 36.46  | 85.00  | **62.42**  | **88.88**  |

Tabel 4：LLaMA-3-8B - Comparison on QA benchmark with training-free methods

| $L_p/L_t$ | Method                  | Ratio  | ARC-c  | ARC-e  | BoolQ  | HeSw   | PIQA   | WG     | WSC    | Race-h | CoPa   | Avg.       | RP         |
|-----------|-------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|------------|------------|
| 0/32      | Dense                   | -      | 53.41  | 77.78  | 81.28  | 79.16  | 80.85  | 72.85  | 86.45  | 40.19  | 89.00  | 73.44      | 100        |
| 7/32      | LLMPruner               | 19.37  | 35.32  | 59.30  | 55.23  | 51.48  | 72.58  | 59.98  | 67.03  | 31.39  | 81.00  | 57.03      | 77.12      |
| 7/32      | SLEB                    | 19.01  | 34.04  | 60.06  | 45.17  | 62.01  | 74.05  | 55.01  | 67.40  | 32.82  | 74.00  | 56.06      | 76.08      |
| 7/32      | ShortGPT                | 19.01  | 42.41  | 56.65  | 65.26  | 64.70  | 70.89  | 71.19  | 73.63  | 34.16  | 75.00  | 61.54      | 83.79      |
| 7/32      | LLM-Streamline (None)   | 19.01  | 28.92  | 39.56  | 38.07  | 33.26  | 59.47  | 55.56  | 59.71  | 24.02  | 60.00  | 44.29      | 59.99      |
| 7/32      | ShortGPT＋LinearPatch    | 18.80  | 43.17  | 60.82  | 75.66  | 66.74  | 72.85  | 70.17  | 75.82  | 37.51  | 77.00  | **64.42**  | **87.82**  |
| 7/32      | LLM-Streamline (None)＋LinearPatch   | 18.80  | 34.39  | 51.26  | 57.52  | 49.31  | 63.33  | 63.22  | 72.53  | 29.95  | 67.00  | 54.28      | 73.57      |


## References

If you find LinearPatch helpful, please cite our paper:

```

@article{chen2025linearpatch,
  title={LinearPatch: Plug-and-Play Patching for Layer-Pruned Large Language Models},
  author={Chen, Xinrui and Bai, Haoli and Bai, and Liu, Ruikang and Zhao, Kang, and Yu, Xianzhi and Hou, Lu and Guan, Tian and He, Yonghong and and Yuan, Chun},
  journal={arXiv preprint arXiv:2505.24680},
  year={2025}
}

```
