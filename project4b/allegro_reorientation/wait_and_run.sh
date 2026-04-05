#!/bin/bash

# --- CONFIGURATION ---
TARGET_PID=$1
FILE_PATH="/home/cc/ee106b/sp26/class/ee106b-aau/106b-sp26-labs-starter/project4b/allegro_reorientation/source/allegro_reorientation/allegro_reorientation/tasks/manager_based/allegro_reorientation/inhand_env_cfg.py"

# Exact strings for matching
OLD_LINE='object_spin_l2 = RewTerm(func=mdp.object_spin_l2, weight=-1e-2, params={"object_cfg": SceneEntityCfg("object")})'
NEW_LINE='object_spin_near_goal_l2 = RewTerm(func=mdp.object_spin_near_goal_l2, weight=-1e-2,params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "angle_threshold": 0.2})'

if [ -z "$TARGET_PID" ]; then
    echo "Error: Please provide a PID. Usage: ./run_experiment.sh <PID>"
    exit 1
fi

# 1. Polling loop
echo "Monitoring PID $TARGET_PID..."
while ps -p $TARGET_PID > /dev/null; do
    sleep 5  # Check every minute
    echo "FAIL once"
done

echo "Process $TARGET_PID has finished. Updating configuration file..."

# 2. Modify the Python file
# Comment out the spin_l2 line
sed -i "s|.*$OLD_LINE|    # $OLD_LINE|" "$FILE_PATH"

# Uncomment the spin_near_goal_l2 line
# This looks for the line even if it has a '#' or spaces in front and removes the comment
sed -i "s|.*#.*$NEW_LINE|    $NEW_LINE|" "$FILE_PATH"

echo "File updated. Launching IsaacLab Training..."

# 3. Run the training command
python scripts/rsl_rl/train.py \
--task Template-Allegro-Reorientation-v0 \
--headless \
--logger wandb \
--log_project_name allegro-reorientation \
--run_name task4_spin_near_goal \
--hand_preset isaaclab