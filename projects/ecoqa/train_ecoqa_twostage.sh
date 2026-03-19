#!/bin/bash
set -euo pipefail
set -x

unset ROCR_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export RAY_TMPDIR=/root/autodl-tmp/ray
export WANDB_DIR=/root/autodl-tmp/wandb
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=10

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-4B-Instruct-2507}
CKPT_DIR=${CKPT_DIR:-/root/autodl-tmp/checkpoints/rllm-agent/ecoqa-4b-twostage}
EXP_NAME=${EXP_NAME:-ecoqa-4b-two-stage}
PROMPT_LEN=${PROMPT_LEN:-2560}
RESP_LEN=${RESP_LEN:-3072}

# Stage-1 defaults (explore + light shaping)
STAGE1_LR=${STAGE1_LR:-5e-6}
STAGE1_ENTROPY=${STAGE1_ENTROPY:-0.0002}
STAGE1_TEMP=${STAGE1_TEMP:-0.7}
STAGE1_SHAPING_ENABLE=${STAGE1_SHAPING_ENABLE:-True}
STAGE1_SHAPING_MAX_BONUS=${STAGE1_SHAPING_MAX_BONUS:-0.1}
# Ensure step 90 is visited/saved when one epoch has 90 internal steps.
STAGE1_TOTAL_STEPS=${STAGE1_TOTAL_STEPS:-91}

# Stage-2 defaults (stabilize + correctness)
STAGE2_LR=${STAGE2_LR:-1e-6}
STAGE2_ENTROPY=${STAGE2_ENTROPY:-0.00005}
STAGE2_TEMP=${STAGE2_TEMP:-0.6}
STAGE2_SHAPING_ENABLE=${STAGE2_SHAPING_ENABLE:-False}
STAGE2_SHAPING_MAX_BONUS=${STAGE2_SHAPING_MAX_BONUS:-0.00}
# Stage-2 runs for one more epoch (another ~90 steps) after resuming from step 90.
# Use 181 so training reaches/saves step 180 with save_freq=10.
STAGE2_TOTAL_STEPS=${STAGE2_TOTAL_STEPS:-181}

VAL_TEMP=${VAL_TEMP:-0.6}

COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_batch_size=8
    data.val_batch_size=8
    data.max_prompt_length=${PROMPT_LEN}
    data.max_response_length=${RESP_LEN}
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.hybrid_engine=True
    actor_rollout_ref.model.lora_rank=16
    actor_rollout_ref.model.lora_alpha=32
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.actor.ppo_mini_batch_size=4
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.val_kwargs.n=8
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMP}
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.max_num_batched_tokens=16384
    actor_rollout_ref.rollout.max_num_seqs=64
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.actor.checkpoint.save_contents=[model]
    actor_rollout_ref.actor.checkpoint.load_contents=[model]
    trainer.max_actor_ckpt_to_keep=2
    trainer.logger=['console','wandb']
    trainer.project_name='rllm-agent'
    trainer.experiment_name=${EXP_NAME}
    trainer.val_before_train=True
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.save_freq=10
    trainer.test_freq=30
    trainer.default_hdfs_dir=null
    trainer.default_local_dir=${CKPT_DIR}
    rllm.agent.max_steps=10
    rllm.workflow.n_parallel_tasks=64
    rllm.mask_truncated_samples=True
    rllm.compact_filtering.enable=True
)

# Stage-1: faster exploration with small shaping bonus.
export ECOQA_ENABLE_SHAPING_BONUS=${STAGE1_SHAPING_ENABLE}
export ECOQA_MAX_SHAPING_BONUS=${STAGE1_SHAPING_MAX_BONUS}
python3 -m projects.ecoqa.train_ecoqa_twostage \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.actor.optim.lr=${STAGE1_LR} \
    actor_rollout_ref.actor.entropy_coeff=${STAGE1_ENTROPY} \
    actor_rollout_ref.rollout.temperature=${STAGE1_TEMP} \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${STAGE1_TOTAL_STEPS}

# Stage-2: keep no curriculum, reduce exploration and close with correctness objective.
export ECOQA_ENABLE_SHAPING_BONUS=${STAGE2_SHAPING_ENABLE}
export ECOQA_MAX_SHAPING_BONUS=${STAGE2_SHAPING_MAX_BONUS}
python3 -m projects.ecoqa.train_ecoqa_twostage \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.actor.optim.lr=${STAGE2_LR} \
    actor_rollout_ref.actor.entropy_coeff=${STAGE2_ENTROPY} \
    actor_rollout_ref.rollout.temperature=${STAGE2_TEMP} \
    trainer.resume_mode=auto \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${STAGE2_TOTAL_STEPS}
