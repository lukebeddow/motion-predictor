# Variables
IMAGE_NAME = motion-predictor
CONTAINER_NAME = motion-predictor
WORKDIR = $(PWD)/src
CONFDIR = $(PWD)/configs
ENV_FLAG = main
ENVPATH = /envs/$(ENV_FLAG)

PORT = 8888
GPU_FLAG = --gpus device=0


# Default target
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make build           - Build the Docker image"
	@echo "  make run             - Run the container with GPU support"
	@echo "  make shell           - Open an interactive shell inside the container"
	@echo "  make jupyter         - Run Jupyter Notebook with GPU support"
	@echo "  make stop            - Stop the running container"
	@echo "  make clean           - Remove the Docker image and clean up"
	
# Build the Docker image
.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .

# Run the container with GPU support
.PHONY: run
run:
	docker run --rm $(GPU_FLAG) -v $(WORKDIR):/workspace -v $(CONFDIR):/workspace/configs $(IMAGE_NAME)

# Open an interactive shell in the container
.PHONY: shell
shell:
	docker run -it --rm $(GPU_FLAG) -v $(WORKDIR):/workspace/src/ -v $(CONFDIR):/workspace/configs $(IMAGE_NAME) \
	bash -c "source $(ENVPATH)/bin/activate && exec bash"

# Run Jupyter Notebook with GPU support 
.PHONY: jupyter
jupyter:
	docker run --rm $(GPU_FLAG) -p $(PORT):8888 -v $(WORKDIR):/workspace $(IMAGE_NAME) \
	bash -c "source $(ENVPATH)/bin/activate && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''  --NotebookApp.password='' "

# Stop the running container (if detached mode is used in future)
.PHONY: stop
stop:
	-docker stop $(CONTAINER_NAME)

# Clean up the Docker image
.PHONY: clean
clean:
	-docker rmi $(IMAGE_NAME)