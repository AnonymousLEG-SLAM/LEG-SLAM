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

## Requirements

The following requirements are essential for building and running LEG-SLAM:

- **CUDA Toolkit 11.8** - Required for GPU acceleration and CUDA compilation
- **NVCC (NVIDIA CUDA Compiler)** - Required for compiling CUDA extensions (`nvcc --version` to verify)
- **PyTorch** - PyTorch 2.0.1 with CUDA 11.8 support (libtorch C++ API)
- **ONNX Model Files** - Required ONNX model files:
  - `dinov2.onnx` - DINOv2 feature extraction model
  - `pca_text_emb64_imagenet.onnx` or `pca_text_emb64_scannet.onnx` - PCA compressor models

## 🔬 Methodology

1. **DINOv2 Feature Extraction**: Extracts rich, self-supervised embeddings from RGB frames.
2. **Talk2DINO Language Alignment**: Transforms **text queries** into DINOv2-compatible feature space.
3. **PCA Compression**: Reduces embeddings from **768D → 64D**, enabling real-time processing.
4. **3D Gaussian Splatting**: Constructs a continuous, high-resolution 3D scene representation.
5. **Semantic Querying**: Computes **cosine similarity** between scene embeddings and textual queries, generating **semantic heatmaps**.

## Setup and Building

1. Install dependencies and required libraries:
```bash
./setup.sh
```


2. Build the project and its dependencies:
```bash
./build.sh
```

## Running the Project

### Example Usage

To run the RGBD-SLAM system with Replica dataset:

```bash
./bin/replica_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    ./cfg/encoder/Replica/room0.yaml \
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

### Note
Make sure to replace `path/to/Replica/office0/` with the actual path to your Replica dataset.


## Object Rendering Example

To train a scene (if not already trained) and render images of a target object using a text request, use the following command:

```bash
python eval/render_object.py --scene_path path/to/Replica/office0 --text_request "trash bin" --encoder_path cfg/encoder/pca_encoder_imagenet.yaml
```

This will:
- Train the scene if it hasn't been trained yet (using the raw frames in the specified directory).
- Render orbit videos of the detected object(s) matching the text request (e.g., "trash bin").
- Output videos will be saved in the `ovs_videos` directory with a name based on the scene and request.

## Docker API Server

The project includes a FastAPI server that can handle both object finding and LEGS-SLAM processing:

```bash
# Start the API server
docker-compose up

# Or run in background
docker-compose up -d
```

The API server will be available at `http://localhost:8005`

### API Endpoints:

- **GET /** - Health check
- **GET /health** - Detailed health status  
- **POST /find_objects** - Find objects in a scene
- **POST /run_legs_slam** - Run LEGS-SLAM processing (with sensible defaults)

### Example API Usage:

```python
import requests

# Find objects in a scene
response = requests.post("http://localhost:8005/find_objects", json={
    "scene_path": "/workspace/data/Replica/office0",
    "prompt": "chair",
    "use_rerun": False,
    "visualize_trajectory": True
})

result = response.json()
print(f"Found {len(result['video_paths'])} video(s)")
for video_path in result['video_paths']:
    print(f"Video: {video_path}")

# Run LEGS-SLAM processing
response = requests.post("http://localhost:8005/run_legs_slam", json={
    "sequence_path": "/workspace/data/Replica/office0/",
    "output_path": "results/office0"
})

result = response.json()
print(f"LEGS-SLAM completed: {result['message']}")
print(f"Output saved to: {result['output_path']}")
```

### Default Parameters for LEGS-SLAM:

The API uses these default parameters:
- **vocabulary_path**: `./ORB-SLAM3/Vocabulary/ORBvoc.txt`
- **orb_settings_path**: `./cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml`
- **encoder_settings_path**: `./cfg/encoder/pca_encoder_scannet.yaml`
- **gaussian_settings_path**: `./cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml`
- **output_path**: `results/{SCENE_NAME}` (replace with actual scene name)

You only need to specify `sequence_path` for most use cases!

### Test the API:

```bash
# Run the test script
python3 test_api.py
```

### Docker Build Issues and Solutions

If you encounter CUDA version mismatches or compilation errors during the Docker build, the following fixes have been implemented:

1. **CUDA Version Compatibility**: PyTorch is automatically reinstalled with CUDA 11.8 support
2. **CUDA Architecture Detection**: Explicit CUDA architectures are specified for compatibility
3. **Package Dependencies**: Missing package directories are automatically created
4. **xFormers Compatibility**: xFormers is reinstalled with PyTorch 2.0.1+cu118 support

The Docker container includes all necessary fixes for:
- CUDA 11.8 compatibility
- PyTorch 2.0.1 support
- CUDA extension compilation
- Package installation issues

## 📜 Paper

If you find this work useful, please cite:

```bibtex
@article{LEG-SLAM,
  title={LEG-SLAM: Open-Vocabulary 3D Gaussian Splatting for SLAM},
  author={Anonymous Authors},
  journal={Submitted ICRA 2026},
  year={2025}
}
```
## ⏳ Code Availability

🔹 Stay tuned for updates! 🚀
