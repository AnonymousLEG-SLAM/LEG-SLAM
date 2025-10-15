# LEGS-SLAM

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
