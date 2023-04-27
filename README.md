# Getting Started

The `Dockerfile` located at `containers/Dockerfile` provides instructions for setting up the build environment. Since LibTorch is a large library, the Dockerfile assumes that the file `libtorch-cxx11-abi-shared-with-deps-*.zip` is present in the directory.

The Dockerfile utilizes an image with CUDA and CuDNN preinstalled, both of which are required to compile the project with GPU support. If GPU support is not necessary, CUDA and CuDNN installations are not needed, and the CPU version of LibTorch is sufficient.

Execute the following command to build and copy the executables `plan` and `train` to the output directory `out`, along with the LibTorch library (with CUDA support). This setup can likely run on other machines outside of a Docker or Apptainer environment:

`sudo docker build -t relational -f containers/Dockerfile . --output out`

Refer to the `containers/cuda_cudnn/Dockerfile` for guidance on installing CUDA and CuDNN on your local machine.

We provide two containers that adhere to the IPC 2023 Learning Track interface. The first container learns from the provided instances:

`sudo apptainer build muninn.sif Apptainer.muninn.learn`

The second container generates a plan using the learned knowledge collected by Muninn:

`sudo apptainer build huginn.sif Apptainer.muninnn.plan`
