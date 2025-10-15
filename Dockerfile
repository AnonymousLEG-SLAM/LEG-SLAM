# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    libeigen3-dev \
    libboost-all-dev \
    libjsoncpp-dev \
    libopengl-dev \
    mesa-utils \
    libglfw3-dev \
    libglm-dev \
    libssl-dev \
    libffi-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    libtbb2 \
    libtbb-dev \
    libdc1394-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for build
ENV OPENCV_VERSION=4.7.0
ENV ONNX_RUNTIME_VERSION=1.17.3
ENV TORCH_VERSION=2.0.1

# Download and build OpenCV with CUDA support
RUN mkdir -p opencv-build && cd opencv-build && \
    wget https://codeload.github.com/opencv/opencv_contrib/tar.gz/refs/tags/${OPENCV_VERSION} -O opencv_contrib-${OPENCV_VERSION}.tar.gz && \
    wget https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/${OPENCV_VERSION} -O opencv-${OPENCV_VERSION}.tar.gz && \
    tar -xzf opencv-${OPENCV_VERSION}.tar.gz && \
    tar -xzf opencv_contrib-${OPENCV_VERSION}.tar.gz && \
    cd opencv-${OPENCV_VERSION} && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE \
          -DWITH_CUDA=ON \
          -DWITH_CUDNN=OFF \
          -DOPENCV_DNN_CUDA=OFF \
          -DWITH_NVCUVID=ON \
          -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
          -DOPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-${OPENCV_VERSION}/modules" \
          -DBUILD_TIFF=ON \
          -DBUILD_ZLIB=ON \
          -DBUILD_JASPER=ON \
          -DBUILD_CCALIB=ON \
          -DBUILD_JPEG=ON \
          -DWITH_FFMPEG=ON \
          -DCMAKE_INSTALL_PREFIX=/usr/local/opencv \
          .. && \
    make -j$(nproc) && \
    make install && \
    cd /workspace && rm -rf opencv-build


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
bash /tmp/miniconda.sh -b -p /opt/conda && \
rm /tmp/miniconda.sh


ENV PATH="/opt/conda/bin:$PATH"

COPY environment.yml /workspace/environment.yml
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
conda env create -f /workspace/environment.yml

SHELL ["conda", "run", "-n", "legs-slam", "/bin/bash", "-c"]


COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
# Reinstall PyTorch with CUDA 11.8 support to fix version mismatch
RUN pip uninstall torch torchvision torchaudio -y && \
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
COPY eval/submodules/ /workspace/pip-submodules/
RUN pip install -e /workspace/pip-submodules/diff-gaussian-rasterization-legs-slam/
RUN pip install -e /workspace/pip-submodules/simple-knn/
    

# Download ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_RUNTIME_VERSION}.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-${ONNX_RUNTIME_VERSION}.tgz && \
    rm onnxruntime-linux-x64-gpu-${ONNX_RUNTIME_VERSION}.tgz

COPY onnxruntime_share onnxruntime-linux-x64-gpu-${ONNX_RUNTIME_VERSION}/share

# Download PyTorch (libtorch)
RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu118.zip -O libtorch-cu118-${TORCH_VERSION}.zip && \
    unzip -q libtorch-cu118-${TORCH_VERSION}.zip && \
    rm libtorch-cu118-${TORCH_VERSION}.zip

# Copy only ORB-SLAM3 for building dependencies
COPY ORB-SLAM3/ /workspace/ORB-SLAM3/

# Set environment variables for the project
ENV OpenCV_DIR=/usr/local/opencv/lib/cmake/opencv4
ENV Torch_DIR=/workspace/libtorch/share/cmake/Torch
ENV onnxruntime_DIR=/workspace/onnxruntime-linux-x64-gpu-${ONNX_RUNTIME_VERSION}
ENV onnxruntime_LIBRARY=/workspace/onnxruntime-linux-x64-gpu-${ONNX_RUNTIME_VERSION}/lib/libonnxruntime.so.${ONNX_RUNTIME_VERSION}
ENV CMAKE_PREFIX_PATH=/workspace/onnxruntime-linux-x64-gpu-${ONNX_RUNTIME_VERSION}

# Build ORB-SLAM3 dependencies
RUN cd /workspace/ORB-SLAM3/Thirdparty/DBoW2 && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=${OpenCV_DIR} && \
    make -j$(nproc)

RUN cd /workspace/ORB-SLAM3/Thirdparty/g2o && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

RUN cd /workspace/ORB-SLAM3/Thirdparty/Sophus && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Build ORB-SLAM3
RUN cd /workspace/ORB-SLAM3/Vocabulary && \
    tar -xf ORBvoc.txt.tar.gz && \
    cd .. && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=${OpenCV_DIR} && \
    make -j$(nproc)

# Copy LEGS-SLAM source code and runtime files
COPY CMakeLists.txt /workspace/
COPY include/ /workspace/include/
COPY viewer/ /workspace/viewer/
COPY cuda_rasterizer/ /workspace/cuda_rasterizer/
COPY third_party/ /workspace/third_party/
COPY examples/ /workspace/examples/
COPY weights/ /workspace/weights/
COPY tools/ /workspace/tools/
COPY encoder_models/ /workspace/encoder_models/
COPY cfg/ /workspace/cfg/
COPY src/ /workspace/src/
COPY eval/ /workspace/eval/

# Build LEGS-SLAM
RUN cd /workspace && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Create a script to run the application
RUN echo '#!/bin/bash\n\
if [ "$#" -ne 6 ]; then\n\
    echo "Usage: $0 <vocabulary_path> <orb_settings_path> <encoder_settings_path> <gaussian_settings_path> <sequence_path> <output_path>"\n\
    echo "Example:"\n\
    echo "  $0 ./ORB-SLAM3/Vocabulary/ORBvoc.txt ./cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml ./cfg/encoder/Replica/room0.yaml ./cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml /path/to/Replica/office0/ ./results/office0"\n\
    exit 1\n\
fi\n\
\n\
./bin/replica_rgbd "$@"\n\
' > /workspace/run_legs_slam.sh && \
    chmod +x /workspace/run_legs_slam.sh

# RUN pip install git+https://github.com/openai/CLIP.git

# Fix xFormers compatibility with PyTorch 2.0.1+cu118
# RUN pip uninstall xformers -y && \
#     pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118

# Set the default command
CMD ["/bin/bash"] 