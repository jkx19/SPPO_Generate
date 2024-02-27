export HF_HOME="/data/jkx/.cache"

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
OUTDIR="mistral02"


# MODEL="alignment-handbook/zephyr-7b-sft-full"
# OUTDIR="zephyr-sft"

PAIRS=3

# python process_data.py

# python generate.py --model $MODEL --maxlen 1024 --output_dir "generated/$OUTDIR" --pairs $PAIRS


CUDA=0
SCPU=$(($CUDA*7))
PCUDA=$(($CUDA+1))
ECPU=$(($PCUDA*7))
export CUDA_VISIBLE_DEVICES="$CUDA"

taskset -c "$SCPU-$ECPU" python rank.py --output_dir $OUTDIR --pairs $PAIRS --numgpu 8 --gpu $CUDA

# python compute_prob.py



# CUDA=0
# export CUDA_VISIBLE_DEVICES="$CUDA"

# taskset -c 0-36 python rank.py --output_dir $OUTDIR --pairs 3 --numgpu 2 --gpu $CUDA

# taskset -c 10-20 python rank.py --output_dir $OUTDIR --pairs 3 --numgpu 2 --gpu 0


# python rank.py --output_dir $OUTDIR --multi_thread --pairs 3 --numgpu 8

# python generate.py --model "HuggingFaceH4/zephyr-7b-beta" --data_frac $FRACB --batch_size 16 --output_dir generated/iteration-1
# accelerate launch --main_process_port=29500 --num_processes=1 --num_machines=1 generate.py --model "alignment-handbook/zephyr-7b-dpo-qlora" --dataset "HuggingFaceH4/ultrafeedback_binarized" --split "train_prefs" --batch_size 4 --frac_len 800 --data_frac 0 --output_dir generated/iteration-0
