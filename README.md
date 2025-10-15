# LEG-SLAM: Real-Time Language-Enhanced Gaussian Splatting for SLAM

![Graphical Abstract](https://github.com/user-attachments/assets/a3b034b9-96ea-4695-b42a-59785103993a)

## Overview

LEG-SLAM is an **open-vocabulary** 3D SLAM system that integrates **3D Gaussian Splatting**, **DINOv2 feature extraction**, and **Talk2DINO language grounding** to enable real-time **semantic 3D scene understanding**. Unlike existing methods, LEG-SLAM allows **text-driven** interactive exploration of reconstructed environments **without predefined object categories**.

### 🔹 Key Features:
- **Real-time 3D Reconstruction:** High-fidelity scene reconstruction with **Gaussian Splatting**.
- **Open-Vocabulary Understanding:** Uses **DINOv2** features and **Talk2DINO** to match text queries to visual features.
- **Efficient Feature Compression:** **PCA-based embedding compression** enables low-latency inference.
- **Interactive Scene Queries:** Retrieve **semantic masks** in real-time by specifying objects via text.
- **High-Speed Performance:** Achieves **10 FPS** on **Replica** and **18 FPS** on **ScanNet**, significantly faster than prior methods.

## 🔬 Methodology

1. **DINOv2 Feature Extraction**: Extracts rich, self-supervised embeddings from RGB frames.
2. **Talk2DINO Language Alignment**: Transforms **text queries** into DINOv2-compatible feature space.
3. **PCA Compression**: Reduces embeddings from **768D → 64D**, enabling real-time processing.
4. **3D Gaussian Splatting**: Constructs a continuous, high-resolution 3D scene representation.
5. **Semantic Querying**: Computes **cosine similarity** between scene embeddings and textual queries, generating **semantic heatmaps**.

## 🐳 Docker Setup

### Prerequisites

- Docker with NVIDIA GPU support (nvidia-docker2)
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed on the host machine

### Building and Running the Docker Container

1. **Clone the repository:**
```bash
git clone https://github.com/AnonymousLEG-SLAM/LEG-SLAM.git
cd LEG-SLAM
```

2. **Build the Docker image:**
```bash
docker-compose build
```

This will:
- Install all system dependencies (OpenCV with CUDA, ONNX Runtime, PyTorch)
- Build ORB-SLAM3 and its dependencies
- Build LEG-SLAM with CUDA support
- Set up the conda environment with all Python dependencies

3. **Run the Docker container:**

**Option A: Run the API server (recommended for easy access)**
```bash
docker-compose up
```

The API server will be available at `http://localhost:8005`

To run in background:
```bash
docker-compose up -d
```

**Option B: Run interactively**
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/ovs_videos:/workspace/ovs_videos \
  legs-slam-api /bin/bash
```

### Using the Docker Container

#### Running LEG-SLAM via API (Easiest Method)

Once the container is running, you can use the REST API:

```python
import requests

# Run LEG-SLAM processing
response = requests.post("http://localhost:8005/run_legs_slam", json={
    "sequence_path": "/workspace/data/Replica/office0/",
    "output_path": "results/office0"
})

result = response.json()
print(f"LEG-SLAM completed: {result['message']}")
print(f"Output saved to: {result['output_path']}")

# Find objects in a scene
response = requests.post("http://localhost:8005/find_objects", json={
    "scene_path": "/workspace/data/Replica/office0",
    "prompt": "chair",
    "use_rerun": False,
    "visualize_trajectory": True
})

result = response.json()
print(f"Found {len(result['video_paths'])} video(s)")
```

#### Running LEG-SLAM via Command Line

Inside the container (or via `docker exec`):

```bash
# Activate the conda environment
conda activate legs-slam

# Run LEG-SLAM
./bin/replica_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    ./cfg/encoder/pca_encoder_scannet.yaml \
    ./cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /workspace/data/Replica/office0/ \
    results/office0
```

#### Finding Objects with Text Queries

```bash
# Inside the container
conda activate legs-slam

python eval/render_object.py \
    --scene_path /workspace/data/Replica/office0 \
    --text_request "trash bin" \
    --encoder_path cfg/encoder/pca_encoder_imagenet.yaml
```

Videos will be saved to `/workspace/ovs_videos/` (mounted to `./ovs_videos` on host).

### Volume Mounts

The Docker container uses the following volume mounts:

- `./data:/workspace/data:ro` - Your dataset directory (read-only)
- `./results:/workspace/results` - Output results directory
- `./ovs_videos:/workspace/ovs_videos` - Object visualization videos
- `./cfg:/workspace/cfg:ro` - Configuration files (read-only)

Make sure to place your datasets in the `./data` directory before running the container.

### API Endpoints

- **GET /** - Health check
- **GET /health** - Detailed health status  
- **POST /find_objects** - Find objects in a scene
- **POST /run_legs_slam** - Run LEG-SLAM processing

### Testing the Setup

```bash
# Run the API test script
python3 test_api.py
```

## 🛠️ Native Build (Alternative to Docker)

If you prefer to build natively instead of using Docker:

### Setup and Building

1. Install dependencies and required libraries:
```bash
./setup.sh
```

2. Build the project and its dependencies:
```bash
./build.sh
```

### Running the Project

To run the RGBD-SLAM system with Replica dataset:

```bash
./bin/replica_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    ./cfg/encoder/pca_encoder_scannet.yaml \
    ./cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    path/to/Replica/office0/ \
    results/office0
```

### Command Line Arguments

1. `path_to_vocabulary`: Path to ORB vocabulary file
2. `path_to_ORB_SLAM3_settings`: Path to ORB-SLAM3 configuration file
3. `path_to_language_features_encoder_setting`: Path to encoder configuration
4. `path_to_gaussian_mapping_settings`: Path to Gaussian mapping configuration
5. `path_to_sequence`: Path to input data sequence
6. `path_to_trajectory_output_directory`: Directory for saving results

## 📜 Paper

If you find this work useful, please cite:

```bibtex
@article{LEG-SLAM,
  title={LEG-SLAM: Open-Vocabulary 3D Gaussian Splatting for SLAM},
  author={Anonymous Authors},
  journal={Under Review at ICCV 2025},
  year={2025}
}
```

## 📄 License

LEG-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

## ⏳ Code Availability

🔹 Full code and models are available in this repository. 🚀