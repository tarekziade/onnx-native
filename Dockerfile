FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive


# Install core packages you might need
RUN apt-get update && apt-get install -y --no-install-recommends \
  wget \
  tar \
  clang \
  build-essential \
  zlib1g-dev \
  libssl-dev \
  libncurses5-dev \
  libffi-dev \
  libsqlite3-dev \
  libreadline-dev \
  libtk8.6 \
  libgdbm-dev \
  uuid-dev \
  libbz2-dev \
  git \
  libprotobuf-dev \
  protobuf-compiler \
  && rm -rf /var/lib/apt/lists/*

# Download and unpack a newer CMake (example: 3.28.2).
# Update the version URL as needed from https://github.com/Kitware/CMake/releases
RUN wget --no-check-certificate https://github.com/Kitware/CMake/releases/download/v3.28.2/cmake-3.28.2-linux-x86_64.tar.gz \
  && tar xzf cmake-3.28.2-linux-x86_64.tar.gz -C /opt \
  && rm cmake-3.28.2-linux-x86_64.tar.gz

# Put this newer CMake ahead of any older system CMake in PATH
ENV PATH="/opt/cmake-3.28.2-linux-x86_64/bin:${PATH}"

# Optionally verify your new CMake version
RUN cmake --version

# 2) Download Python 3.12.0 source from python.org (adjust version as needed)
RUN wget --no-check-certificate https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz \
  && tar xzf Python-3.12.0.tgz \
  && cd Python-3.12.0 \
  # 3) Configure & build (use --enable-optimizations if desired, but be aware it increases build time)
  && ./configure --enable-optimizations \
  && make -j$(nproc) \
  # 4) 'altinstall' so we don’t overwrite the system python3
  && make altinstall \
  # 5) Clean up source
  && cd .. \
  && rm -rf Python-3.12.0 Python-3.12.0.tgz

ENV USE_SYSTEM_PYTHON=/usr/local/bin/python3.12

RUN apt-get update && apt-get install -y lsb-release software-properties-common gnupg

RUN wget --no-check-certificate -qO- https://apt.llvm.org/llvm.sh | bash -s -- 18
RUN clang-18 --version

RUN wget --no-check-certificate https://bootstrap.pypa.io/get-pip.py
RUN /usr/local/bin/python3.12 get-pip.py


# Set clang as default compiler
ENV CC=clang-18
ENV CXX=clang++-18

WORKDIR /workspace
CMD ["/bin/bash"]
