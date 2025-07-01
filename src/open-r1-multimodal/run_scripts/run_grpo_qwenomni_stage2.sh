ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

RUN_NAME="stage2"
export LOG_PATH="./debug_log_$RUN_NAME.txt"




export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=INFO

mkdir -p output/$RUN_NAME
cp $0  output/$RUN_NAME
torchrun  --nproc_per_node 8 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo_qwenomni.py \
    --deepspeed run_scripts/zero3_offload.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path $STAGE1 \
    --dataset_name data_config/stage2.yaml \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --freeze_vision_modules true \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --scale_rewards false \
    --reward_funcs  format accuracy reasoning context \
    --use_audio_in_video true \
    --gradient_checkpointing true \
    --log_completions true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model false  2>&1 | tee output/$RUN_NAME/train.log


exit ${PIPESTATUS[0]}