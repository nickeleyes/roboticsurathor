# Docker build changes

This note explains the Docker build changes made for ARM64 machines, especially NVIDIA Jetson/Orin-style hosts.

## Problem

The deploy image is built from:

```dockerfile
FROM nvgear/ros-2:latest
```

Before this change, the build flow could resolve `nvgear/ros-2:latest` from the remote registry. That remote image was built for `linux/amd64`, so ARM64 machines could end up trying to build the deploy container on top of an AMD64 base image.

That caused architecture mismatch problems on ARM64.

## Base image build

Previously, `build_deploy_base.sh` was set up to build and push an AMD64 base image:

```bash
sudo docker buildx build \
    --platform linux/amd64 \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_NAME}:${TAG}" \
    --push \
    .
```

The primary issue is that the existing docker file defaulted on building an image based on the linux amd64 platform. The problem with this is that the Jetson Thor system is actually an ARM-64 CPU architecture. The existing code did indeed have a commented out region - so to resolve the issue you must use that. MAin problem: the nvgear/ros-2:latest - is the deploy image built from location, but that uses an incorrect image that exists on the network server elsewhere. THat one that exists elsewhere is exclusively constructed for AMD-64, which means everytime the run.docker file refers to nvgear/ros-2:latest, even if the file did indeed build a new image based on arm64, it overrides that image / ignores it by refering to the online version - therefore pull=false resolves this issue.


Now the script builds an ARM64 base image and loads it into the local Docker image store:

```bash
sudo docker buildx build \
    --platform linux/arm64 \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_NAME}:${TAG}" \
    --load \
    .
```

The important part is `--load`: it makes the locally built ARM64 image available to normal Docker builds under the same tag, `nvgear/ros-2:latest`.

## Deploy image build

Previously, `run_docker.sh` built the deploy image with Buildx and cache metadata:

```bash
sudo docker buildx build \
    --build-arg USERNAME=$USERNAME \
    --build-arg USERID=$USERID \
    --build-arg HOME_DIR=$DOCKER_HOME_DIR \
    --build-arg WORKTREE_NAME=$WORKTREE_NAME \
    --cache-from $CACHE_FROM \
    -t $DEPLOY_CONTAINER \
    -f docker/Dockerfile.deploy \
    --load \
    .
```

Now it uses the regular Docker builder and disables pulling:

```bash
sudo docker build --pull=false \
    --build-arg USERNAME=$USERNAME \
    --build-arg USERID=$USERID \
    --build-arg HOME_DIR=$DOCKER_HOME_DIR \
    --build-arg WORKTREE_NAME=$WORKTREE_NAME \
    --tag $DEPLOY_CONTAINER \
    -f docker/Dockerfile.deploy \
    .
```

The important part is `--pull=false`: Docker should use the locally available `nvgear/ros-2:latest` base image instead of checking the registry and pulling the remote AMD64 image again.

## Build order

The full `--build` flow in `run_docker.sh` is now:

```bash
install_docker_buildx
install_nvidia_toolkit

sudo bash "$SCRIPT_DIR/build_deploy_base.sh"

build_docker_image
```

So the sequence is:

1. Ensure Docker Buildx is installed.
2. Ensure the NVIDIA Container Toolkit is installed.
3. Build and locally load the ARM64 `nvgear/ros-2:latest` base image.
4. Build the project deploy image from that local ARM64 base image.

In short, the change makes the ARM64 base image explicit and local before the deploy image is built. This avoids accidentally reusing or pulling the remote AMD64 base image during the deploy build.

## NVIDIA GPU runtime

There was also an important runtime change for GPU access inside the container.

Previously, ARM64 tried to expose GPU device files directly:

```bash
GPU_RUNTIME_ARGS="--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-modeset --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools"
```

That can make some device nodes visible, but it does not fully configure the container with the NVIDIA driver libraries and runtime hooks. In practice, the container could fail to use CUDA/NVIDIA acceleration correctly and fall back to CPU execution, which made the workload much slower.

Now ARM64 uses the NVIDIA container runtime:

```bash
if is_arm64; then
    GPU_RUNTIME_ARGS="--runtime=nvidia"
else
    GPU_RUNTIME_ARGS="--gpus all --runtime=nvidia"
fi
```

The script also installs/configures the NVIDIA Container Toolkit during build setup:

```bash
install_nvidia_toolkit
```

and passes the NVIDIA runtime environment into `docker run`:

```bash
-e NVIDIA_VISIBLE_DEVICES=all
-e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
-e __GLX_VENDOR_LIBRARY_NAME=nvidia
```

Together, these settings let the container see the NVIDIA GPU and use the host NVIDIA driver stack for compute and graphics. That is what prevents the code from silently running on CPU when GPU acceleration is expected.
