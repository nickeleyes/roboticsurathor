#!/bin/bash
# set -e  # Exit on error

# echo "Dependencies installed successfully. Starting interactive bash shell..."
# exec /bin/bash

#!/bin/bash
set -e

source "${HOME}/venv/bin/activate"
source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=1

cd "${GR00T_WBC_DIR:-$(pwd)}"

# if ! python -c "import tyro" >/dev/null 2>&1; then
#     echo "Python deps missing; installing project deps..."
#     ./docker/entrypoint/install_deps.sh
# fi

echo "Starting interactive bash shell..."
exec /bin/bash