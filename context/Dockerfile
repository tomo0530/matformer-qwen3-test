ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

ARG USERNAME

# ==========================================
# Environmental variables specified in .env file
# ==========================================
ARG USER_UID

# ==========================================
# Python version specified in docker-compose.yml
# ==========================================
ARG PYTHON_MAJOR_VERSION
ARG PYTHON_MINOR_VERSION
ARG PYTHON_PATCH_VERSION
ARG POETRY_HOME="/opt/poetry"
ARG POETRY_VERSION

ENV PYTHON_VERSION ${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}.${PYTHON_PATCH_VERSION}

# ==========================================
# Set timezone
# ==========================================
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ==========================================
# install Python
# ==========================================
# runtime dependencies

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    netbase \
    tzdata \
    dpkg-dev \
    gcc \
    gnupg dirmngr \
    libbluetooth-dev \
    libbz2-dev \
    libc6-dev \
    libdb-dev \
    libexpat1-dev \
    libffi-dev \
    libgdbm-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    make \
    curl \
    tk-dev \
    uuid-dev \
    wget \
    xz-utils \
    zlib1g-dev \
    git \
    openssh-client \
    zsh \
    zip \
    unzip \
    bash-completion \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" \
    && tar xJf python.tar.xz \
    && rm python.tar.xz

WORKDIR /Python-${PYTHON_VERSION}

RUN ./configure \
    && make \
    && make install

WORKDIR /

# ==========================================
# install pip and change symbolic link
# ==========================================
RUN add-apt-repository universe \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev\
    && rm -rf /usr/bin/python \
    && ln -s /usr/local/bin/python${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION} /usr/bin/python \
    && python -m pip install --no-cache-dir pip --force-reinstall \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# ==========================================
# poetry
# ==========================================
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -sSL https://install.python-poetry.org/ | python3 - --version ${POETRY_VERSION} && \
    ln -s ${POETRY_HOME}/bin/poetry /usr/local/bin/poetry


# ==========================================
# Create user in the container to avoid permission matter 
# incompatible between host and container user
# ==========================================
RUN useradd --uid ${USER_UID} -m ${USERNAME} 

USER $USERNAME

WORKDIR /home/$USERNAME/