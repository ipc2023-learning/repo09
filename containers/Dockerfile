# Use our own CuDNN and CUDA image
# FROM cuda_cudnn

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04 AS build_stage

ENV DEBIAN_FRONTEND noninteractive

# Install libtorch
RUN apt-get update && apt-get install -y unzip wget
# RUN wget --no-verbose https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip
COPY libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip .
# COPY libtorch-cxx11-abi-shared-with-deps-2.0.0+cu117.zip .
RUN unzip libtorch*zip -d /usr/local/
RUN rm libtorch*zip

# Compile neural network
RUN apt-get install -y g++ libboost-dev cmake gcc-multilib g++-multilib
COPY CMakeLists.txt /container/CMakeLists.txt
COPY src /container/src
RUN mkdir -p /container/build/ && cd /container/build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -- -j 8

ENTRYPOINT [ "/container/build/src/train" ]

# Export artifacts

FROM scratch AS export_stage

COPY --from=build_stage /usr/local/libtorch/lib lib
COPY --from=build_stage /container/build/src/train .
COPY --from=build_stage /container/build/src/plan .
