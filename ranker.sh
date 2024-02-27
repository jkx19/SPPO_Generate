# export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
export HF_HOME="/data/jkx/.cache"

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
OUTDIR="mistral"
PAIRS=3

CUDA=7
SCPU=$(($CUDA*7))
PCUDA=$(($CUDA+1))
ECPU=$(($PCUDA*7))
export CUDA_VISIBLE_DEVICES="$CUDA"

taskset -c "$SCPU-$ECPU" python rank.py --output_dir $OUTDIR --pairs 3 --numgpu 8 --gpu $CUDA
