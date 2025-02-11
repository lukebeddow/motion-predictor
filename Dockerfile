FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYENV_ROOT="/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
ENV PATH="/.venv/bin:$PATH"

# Set the working directory
WORKDIR /workspace

# Install system dependencies (maybe can reduce these)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        unzip \
        wget \
        git \
        software-properties-common \
        libgl1-mesa-dev \
        libgl1-mesa-glx \
        libosmesa6-dev \
        libglfw3 \
        ffmpeg \
        make \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash && \
    pyenv update && \
    rm -rf $PYENV_ROOT/.git

# Install Python and set as global version
RUN pyenv install 3.11 && \
    pyenv global 3.11

# Set up virtual environment
RUN python -m venv /.venv && \
    pip install --upgrade pip setuptools wheel

# Copy requirements.txt into the container
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install -r requirements.txt

# Setup entry point to activate virtual environment
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]


CMD ["bash"]