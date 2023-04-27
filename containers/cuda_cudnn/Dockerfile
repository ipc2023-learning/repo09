FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# Install CUDA
RUN apt-get update \
    && apt-get install -y wget

# RUN wget --no-verbose https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
COPY cuda-ubuntu2004.pin .
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# RUN wget --no-verbose https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb
COPY cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb .
RUN apt-get install -y gnupg \
    && dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb \
    && apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub \
    && apt-get update

RUN apt-get install -y cuda cuda-toolkit-11-6

RUN rm cuda-repo-ubuntu2004-*.deb

# Install CuDNN
COPY cudnn-local-repo-ubuntu2004-8.4.1.50_1.0-1_amd64.deb .
RUN dpkg -i cudnn-local-repo-ubuntu2004-*_amd64.deb \
    && cp /var/cudnn-local-repo-ubuntu2004-8.4.1.50/cudnn-local-E3EC4A60-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && apt-get install -y libcudnn8-dev \
    && rm cudnn-local-repo-ubuntu2004-*_amd64.deb


# When using --squash:
# RUN apt-get purge -y wget gnupg (remove libcudnn8-dev?)
# RUN apt-get clean autoclean
# RUN rm -rf /var/lib/{apt,dpkg,cache,log}/
