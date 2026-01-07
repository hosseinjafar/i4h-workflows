# Allow Docker to access X11 display
xhost +local:docker
# HOST_HOME=${HOME} # for local pc
HOST_HOME=/Data/hosseinj # for proton

# Run container with GUI support
docker run --name isaac-sim -it --gpus all --rm \
    --network=host \
    --runtime=nvidia \
    --entrypoint=bash \
    -e DISPLAY=$DISPLAY \
    -e "OMNI_KIT_ACCEPT_EULA=Y" \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOST_HOME/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v $HOST_HOME/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v $HOST_HOME/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v $HOST_HOME/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v $HOST_HOME/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v $HOST_HOME/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v $HOST_HOME/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v $HOST_HOME/docker/isaac-sim/documents:/root/Documents:rw \
    -v $HOST_HOME/.cache/i4h-assets:/root/.cache/i4h-assets:rw \
    -v $HOST_HOME/.cache/huggingface:/root/.cache/huggingface:rw \
    -v $HOST_HOME/docker/rti:/root/rti:ro \
    robotic_us:latest
