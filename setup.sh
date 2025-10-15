echo "Installing dependencies..."
sudo apt install libeigen3-dev libboost-all-dev libjsoncpp-dev libopengl-dev mesa-utils libglfw3-dev libglm-dev


if [ ! -d "opencv-build" ]; then
    echo "Downloading opencv..."
    mkdir -p opencv-build
    cd opencv-build

    wget https://codeload.github.com/opencv/opencv_contrib/tar.gz/refs/tags/4.7.0 -O opencv_contrib-4.7.0.tar.gz
    wget https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/4.7.0 -O opencv-4.7.0.tar.gz
    tar -xzf opencv-4.7.0.tar.gz
    tar -xzf opencv_contrib-4.7.0.tar.gz

    cd opencv-4.7.0
    mkdir -p build
    cd build
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON -DWITH_NVCUVID=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 -DOPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-4.7.0/modules" \
    -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_JASPER=ON -DBUILD_CCALIB=ON -DBUILD_JPEG=ON -DWITH_FFMPEG=ON -DCMAKE_INSTALL_PREFIX=../../opencv_install ..

    make -j 16
    make install

    cd ../../../
fi


if [ ! -d "onnxruntime-linux-x64-gpu-1.17.3" ]; then
    echo "Downloading ONNX Runtime..."
    echo "${PWD}"
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-gpu-1.17.3.tgz
    tar -xzf onnxruntime-linux-x64-gpu-1.17.3.tgz
    
    cp -r onnxruntime_share onnxruntime-linux-x64-gpu-1.17.3/share
fi


if [ ! -d "libtorch" ]; then
    echo "Downloading libtorch..."
    wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip -O libtorch-cu118-2.0.1.zip
    unzip -q libtorch-cu118-2.0.1.zip
fi
