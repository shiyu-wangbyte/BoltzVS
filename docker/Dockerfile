# BoltzVS — self-contained image.
#
# Build:
#   docker build -t boltzvs:latest  -f docker/Dockerfile .
#   docker build -t boltzvs:kernels -f docker/Dockerfile --build-arg WITH_KERNELS=1 .
#
# Behind a slow or blocked docker.io, point BASE at a mirror:
#   --build-arg BASE=docker.m.daocloud.io/nvidia/cuda:12.8.0-base-ubuntu22.04
# That mirror was checked against the official registry and serves the identical
# image -- both report sha256:12242992c121f6cab0ca11bccbaaf757db893b3065d7db74b933e59f321b2cf4.
# A digest is content-addressed, so a matching one cannot hide a changed image.
#
# Run (the weight cache is a volume, never baked in -- see below). The entrypoint is
# `boltz`, so the first argument is the subcommand:
#   docker run --gpus all -v boltz_cache:/cache -v "$PWD":/work -w /work boltzvs:latest \
#     screen --receptor r.fasta --library lib.smi --msa r_0.csv --out-dir run/
#
# The ~7 GB of Boltz-2 checkpoints and chemical-component data are NOT in the image.
# They download into /cache on first use and persist in the named volume, so only the
# first run pays for them. Baking them in would quadruple the image and pin a weight
# version to it.
ARG BASE=nvidia/cuda:12.8.0-base-ubuntu22.04
FROM ${BASE}

# triton compiles a CUDA shim at run time, so the image needs a C compiler *and*
# the Python headers: its cuda_utils.c includes Python.h. Without python3.10-dev
# everything looks fine -- import torch, torch.cuda.is_available(), --help all pass
# -- and the first prediction dies in gcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Ubuntu 22.04 ships pip 22, whose isolated build env picks a setuptools older than
# the 64 that PEP 660 needs -- an editable install fails there. Upgrade first.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Torch first and on its own layer: it is most of the image and changes least,
# so edits to BoltzVS do not re-download 2.5 GB.
RUN pip install --no-cache-dir torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Run as the host user to keep the outputs yours:
#   docker run --user "$(id -u):$(id -g)" ...
# That uid owns no home and cannot write site-packages, so every library that
# caches compiled code needs a writable directory of its own. numba defaults to
# caching next to its source and fails outright without this.
# Kept after the torch layer on purpose: editing these must not re-download 2.5 GB.
ENV PYTHONUNBUFFERED=1 \
    BOLTZ_CACHE=/cache \
    HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp/numba \
    TRITON_CACHE_DIR=/tmp/triton \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /opt/boltzvs
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
COPY min_ipae.py ./
# Not editable: nothing edits the source inside an image, and a regular install
# avoids depending on the build backend's PEP 660 support.
RUN pip install --no-cache-dir ".[cuda]"

# The fused trunk kernels are optional and off by default, exactly as in the repo.
# Set WITH_KERNELS=1 to bake them in; they are third-party Apache-2.0 code and the
# kit's own LICENSE and NOTICE travel with the clone under /opt/kernels.
ARG WITH_KERNELS=0
RUN if [ "$WITH_KERNELS" = "1" ]; then \
        boltz install-kernels --cache /opt/kernels && boltz install-kernels --check ; \
    else \
        echo "kernels not installed; run 'boltz install-kernels' inside the container" ; \
    fi

# World-writable so an arbitrary --user uid can use them.
RUN mkdir -p /tmp/numba /tmp/triton /tmp/mpl /cache && chmod 1777 /tmp/numba /tmp/triton /tmp/mpl /cache

VOLUME /cache
WORKDIR /work
ENTRYPOINT ["boltz"]
CMD ["--help"]
