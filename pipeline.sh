export HF_HOME="/data/jkx/.cache"

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
OUTDIR="mistral02"


# MODEL="alignment-handbook/zephyr-7b-sft-full"
# OUTDIR="zephyr-sft"

PAIRS=3

#####################
# Generate Data
#####################

# python generate.py --model $MODEL --maxlen 1024 --output_dir "generated/$OUTDIR" --pairs $PAIRS --world_size 8


CUDA=6
SCPU=$(($CUDA*7))
PCUDA=$(($CUDA+1))
ECPU=$(($PCUDA*7))
export CUDA_VISIBLE_DEVICES="$CUDA"

taskset -c "$SCPU-$ECPU" python rank.py --output_dir $OUTDIR --pairs $PAIRS --numgpu 8 --gpu $CUDA

# python compute_prob.py

