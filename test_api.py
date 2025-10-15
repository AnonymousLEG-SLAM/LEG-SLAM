#!/usr/bin/env python3
"""
Test script for the LEGS-SLAM Object Finder API
"""

import requests

API_BASE_URL = "http://localhost:8005"

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print("Health check response:", response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_find_objects(scene_path: str, prompt: str):
    """Test the find_objects endpoint"""
    try:
        payload = {
            "scene_path": scene_path,
            "prompt": prompt,
            "use_rerun": False,
            "visualize_trajectory": True
        }
        
        print(f"Sending request to find objects with prompt: '{prompt}'")
        print(f"Scene path: {scene_path}")
        
        response = requests.post(f"{API_BASE_URL}/find_objects", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("Success!")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            print(f"Video paths: {result['video_paths']}")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def test_run_legs_slam():
    """Test the run_legs_slam endpoint"""
    try:
        payload = {
            # "sequence_path": "/workspace/data/Replica/office0/",
            "sequence_path": "data/Replica/office1/",
            "output_path": "results/office1"
        }
        
        print("Sending request to run LEGS-SLAM...")
        print(f"Output path: {payload['output_path']}")
        
        response = requests.post(f"{API_BASE_URL}/run_legs_slam", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("Success!")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            print(f"Output path: {result['output_path']}")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def main():
    """Main test function"""
    print("Testing LEGS-SLAM Object Finder API")
    print("=" * 50)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    if not test_health():
        print("❌ Health check failed. Make sure the API server is running.")
        return
    
    print("✅ Health check passed!")
    print()
    
        
    # Test run_legs_slam endpoint
    print("2. Testing run_legs_slam endpoint...")
    
    result = test_run_legs_slam()
    
    if result:
        print("✅ LEGS-SLAM test completed!")
        print(f"Output saved to: {result['output_path']}")
    else:
        print("❌ LEGS-SLAM test failed.")
        
    # Test find_objects endpoint
    print("3. Testing find_objects endpoint...")
    
    # Example scene path and prompt
    # scene_path = "/workspace/data/Replica/office0"  # Adjust this path
    scene_path = "data/Replica/office1"  # Adjust this path
    prompt = "chair"
    
    result = test_find_objects(scene_path, prompt)
    
    if result:
        print("✅ Object finding test completed!")
        print(f"Found {len(result['video_paths'])} video(s)")
        for i, video_path in enumerate(result['video_paths']):
            print(f"  Video {i+1}: {video_path}")
    else:
        print("❌ Object finding test failed.")
    
    print()

if __name__ == "__main__":
    main() 