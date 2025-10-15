# DBoW2
cd ./ORB-SLAM3/Thirdparty/DBoW2
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=../../../../opencv-build/opencv_install/lib/cmake/opencv4
make -j

cd ../../g2o

# g2o
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../Sophus

# Sophus
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# ORB_SLAM3
cd ../../../Vocabulary
echo "Uncompress vocabulary ..."
tar -xf ORBvoc.txt.tar.gz
cd ..

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DOpenCV_DIR=../../opencv-build/opencv_install/lib/cmake/opencv4
make -j

cd ../..

# Legs-SLAM
echo "Building Legs-SLAM ..."
mkdir -p build_debug
cd build_debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j8
