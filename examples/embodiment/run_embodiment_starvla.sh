#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_TYPE=standard
export ROBOT_PLATFORM=LIBERO
export STARVLA_QWENGROOT_SFT_CKPT=/data/checkpoints_rui/starVLA_results/1229_libero3vl4B_qwen3gr00t/checkpoints/steps_30000_pytorch_model.pt

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_NO_OMNI_LOGS=${OMNIGIBSON_NO_OMNI_LOGS:-1}
export OMNIGIBSON_DEBUG=${OMNIGIBSON_DEBUG:-0}
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

if [ -z "$1" ]; then
    CONFIG_NAME=${CONFIG_NAME:-"libero_spatial_ppo_starvla_qwengroot_reinflow"}
else
    CONFIG_NAME=$1
fi

echo "Evaluation Mode: Standard LIBERO"
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"
echo "Using STARVLA_QWENGROOT_SFT_CKPT=$STARVLA_QWENGROOT_SFT_CKPT"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment_starvla.log"
mkdir -p "${LOG_DIR}"

# GPU selection.
# Default to GPU 0 if CUDA_VISIBLE_DEVICES is not already set outside.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Forward all arguments after:
#   $1 = CONFIG_NAME
# Example:
#   bash run_embodiment_starvla.sh config_name runner.max_epochs=1
EXTRA_ARGS=("${@:2}")

CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
    "${EXTRA_ARGS[@]}"
)

printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
echo >> "${MEGA_LOG_FILE}"

"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"

# Use examples:
#   bash run_embodiment_starvla.sh
#   bash run_embodiment_starvla.sh libero_spatial_ppo_starvla_qwengroot_reinflow
#   bash run_embodiment_starvla.sh libero_object_ppo_starvla_qwengroot_reinflow runner.max_epochs=1
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_embodiment_starvla.sh libero_goal_ppo_starvla_qwengroot_reinflow
#   bash run_embodiment_starvla.sh libero_10_ppo_starvla_qwengroot_reinflow runner.only_eval=True runner.val_check_interval=1
