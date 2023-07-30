PROJECT ?= vidar_release
WORKSPACE ?= /workspace/$(PROJECT)
DOCKER_IMAGE ?= ${PROJECT}:latest

SHMSIZE ?= 444G
WANDB_MODE ?= run
DOCKER_OPTS := \
			--name ${PROJECT} \
			--rm -it \
			--shm-size=${SHMSIZE} \
			-e AWS_DEFAULT_REGION \
			-e AWS_ACCESS_KEY_ID \
			-e AWS_SECRET_ACCESS_KEY \
			-e WANDB_API_KEY \
			-e WANDB_ENTITY \
			-e WANDB_MODE \
			-e HOST_HOSTNAME= \
			-e OMP_NUM_THREADS=1 -e KMP_AFFINITY="granularity=fine,compact,1,0" \
			-e OMPI_ALLOW_RUN_AS_ROOT=1 \
			-e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
			-e NCCL_DEBUG=VERSION \
            -e DISPLAY=${DISPLAY} \
            -e XAUTHORITY \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
			-v ~/.aws:/root/.aws \
			-v /root/.ssh:/root/.ssh \
			-v ~/.cache:/root/.cache \
			-v /data/:/data \
			-v /dev/null:/dev/raw1394 \
			-v /mnt/fsx/tmp:/tmp \
			-v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0 \
			-v /var/run/docker.sock:/var/run/docker.sock \
			-v ${PWD}:${WORKSPACE} \
			-w ${WORKSPACE} \
			--privileged \
			--ipc=host \
			--network=host

NGPUS=$(shell nvidia-smi -L | wc -l)

all: clean

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

docker-build:
	docker build \
		-f docker/Dockerfile \
		-t ${DOCKER_IMAGE} .

docker-interactive: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} bash

docker-run: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} bash -c "${COMMAND}"
