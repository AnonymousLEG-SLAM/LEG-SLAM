import argparse
import clip
import cv2
import numpy as np
import os
import shutil
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import time
import torch
from pixelwise_decoder import PixelwiseCompressor
from tqdm import tqdm

def build_text_embedding(categories):
    """Build CLIP text embeddings for given categories."""
    model, _ = clip.load("ViT-L/14@336px")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        all_text_embeddings = []
        print("Building text embeddings...")
        for category in tqdm(categories):
            texts = clip.tokenize(category).to(device)
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)

        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
    return all_text_embeddings.cpu().numpy().T.astype(np.float32)

def resize_and_crop_image(image, desired_size, padded_size, 
                         aug_scale_min=1.0, aug_scale_max=1.0,
                         seed=1, method=tf.image.ResizeMethod.BILINEAR,
                         logarithmic_sampling=False):
    """Resize and pad image to desired size with optional augmentation."""
    with tf.name_scope('resize_and_crop_image_'):
        image_size = tf.cast(tf.shape(image)[0:2], tf.float32)
        random_jittering = (aug_scale_min != 1.0 or aug_scale_max != 1.0)

        if random_jittering:
            if logarithmic_sampling:
                random_scale = tf.exp(tf.random_uniform([],
                                    np.log(aug_scale_min),
                                    np.log(aug_scale_max),
                                    seed=seed))
            else:
                random_scale = tf.random_uniform([],
                                               aug_scale_min,
                                               aug_scale_max,
                                               seed=seed)
            scaled_size = tf.round(random_scale * desired_size)
        else:
            scaled_size = desired_size

        scale = tf.minimum(scaled_size[0] / image_size[0],
                          scaled_size[1] / image_size[1])
        scaled_size = tf.round(image_size * scale)
        image_scale = scaled_size / image_size

        if random_jittering:
            max_offset = scaled_size - desired_size
            max_offset = tf.where(tf.less(max_offset, 0), 
                                tf.zeros_like(max_offset), max_offset)
            offset = max_offset * tf.random_uniform([2], 0, 1, seed=seed)
            offset = tf.cast(offset, tf.int32)
        else:
            offset = tf.zeros((2,), tf.int32)

        scaled_image = tf.image.resize_images(
            image, tf.cast(scaled_size, tf.int32), method=method)

        if random_jittering:
            scaled_image = scaled_image[offset[0]:offset[0] + desired_size[0],
                                      offset[1]:offset[1] + desired_size[1], :]

        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0,
                                                   padded_size[0], padded_size[1])

        image_info = tf.stack([
            image_size,
            tf.constant(desired_size, dtype=tf.float32), 
            image_scale,
            tf.cast(offset, tf.float32)
        ])
        return output_image, image_info

def create_heatmap(sim_matrix, title, crop=None):
    """Create visualization heatmap from similarity matrix."""
    if crop is not None:
        confidence_matrix = (sim_matrix[crop[0]:crop[1]+1, crop[2]:crop[3]+1] + 1) / 2
    else:
        confidence_matrix = (sim_matrix + 1) / 2
    confidence_matrix = np.clip(confidence_matrix, 0, 1)
    confidence_map = (confidence_matrix * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(confidence_map, cv2.COLORMAP_JET)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    text_x = (heatmap.shape[1] - text_size[0]) // 2
    text_y = 30
    cv2.putText(heatmap, title, (text_x, text_y), font, font_scale, (0,0,0), thickness)
    return heatmap
            
def process_frames(frames_dir, embedding_dim, output_dir, model_path, checkpoint_path, debug_prompt, debug_mode):
    """Process video frames to extract and compress embeddings."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    compressor = PixelwiseCompressor(input_dim=768, bottleneck_dim=embedding_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    compressor.load_state_dict(checkpoint['model_state_dict'])
    compressor.eval()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_synchronous_execution(False)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 6)])

    openseg_model = tf2.saved_model.load(model_path, tags=[tf.saved_model.tag_constants.SERVING])
    
    text_embedding = build_text_embedding([debug_prompt])
    text_embedding = tf.reshape(text_embedding, [-1, 1, text_embedding.shape[-1]])
    text_embedding = tf.cast(text_embedding, tf.float32)
    text_embedding_feat = tf.squeeze(text_embedding[0], axis=0)
    
    total_time = 0
    model_time = 0
    frame_count = 0
    
    file_list = [f for f in sorted(os.listdir(frames_dir)) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'depth' not in f]

    for frame_file in tqdm(file_list):
        frame_path = os.path.join(frames_dir, frame_file)
        start_time = time.time()
        
        with tf.io.gfile.GFile(frame_path, 'rb') as f:
            np_image_string = np.array([f.read()])
        
        model_start_time = time.time()  
        with torch.no_grad():  
            output = openseg_model.signatures['serving_default'](
                inp_image_bytes=tf.convert_to_tensor(np_image_string[0]),
                inp_text_emb=text_embedding)
        model_time += time.time() - model_start_time
    
        post_start = time.time()
        image = output['image'].numpy()

        non_zero_mask = np.any(image != 0, axis=2)
        y_indices, x_indices = np.nonzero(non_zero_mask)
        min_y, max_y = y_indices.min(), y_indices.max()
        min_x, max_x = x_indices.min(), x_indices.max()
        post_time = time.time() - post_start
        
        source_load_start = time.time()
        source_image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        target_w, target_h = source_image.shape[1], source_image.shape[0]
        source_load_time = time.time() - source_load_start
            
        preprocess_start = time.time()
        embedding_feat = output['ppixel_ave_feat'].numpy()
        embedding_tensor = torch.from_numpy(embedding_feat[0]).to(device).view(-1, 768)
        preprocess_time = time.time() - preprocess_start
        
        compress_start = time.time()
        with torch.no_grad():
            compressed_padded = compressor.encode(embedding_tensor).view(640, 640, embedding_dim)
        compress_time = time.time() - compress_start
        
        compressed = torch.nn.functional.interpolate(
            compressed_padded[min_y:max_y+1, min_x:max_x+1].unsqueeze(0).permute(0,3,2,1),
            size=(target_w, target_h),
            mode='bilinear'
        ).squeeze(0).permute(2,1,0)

        save_start = time.time()
        compressed_output_path = os.path.join(output_dir, f'{frame_file[:-4]}.npy')
        compressed_np = compressed.cpu().numpy()
        np.save(compressed_output_path, compressed_np)
        save_time = time.time() - save_start
        
        debug_time = 0
        if debug_mode:
            debug_start = time.time()            
            with torch.no_grad():
                decompressed_embedding = compressor.decode(compressed.reshape(target_h, target_w, embedding_dim)).reshape(target_h, target_w, 768).cpu().numpy()
            compressed_similarity = np.tensordot(decompressed_embedding, text_embedding_feat.numpy(), axes=[[2], [0]])
            
            similarity_matrix = tf.tensordot(embedding_feat[0], text_embedding_feat, axes=[[2], [0]])
            
            original_heatmap = create_heatmap(similarity_matrix.numpy(), f"Original Heatmap - '{debug_prompt}'", crop=[min_y, max_y, min_x, max_x])
            compressed_heatmap = create_heatmap(compressed_similarity, f"Compressed Heatmap - '{debug_prompt}'")
            
            original = cv2.imread(frame_path)
            original = cv2.resize(original, (original_heatmap.shape[1], original_heatmap.shape[0]))
            
            if compressed_heatmap.shape != original_heatmap.shape:
                compressed_heatmap = cv2.resize(compressed_heatmap, (original_heatmap.shape[1], original_heatmap.shape[0]))
                
            combined = np.concatenate((original, original_heatmap, compressed_heatmap), axis=1)
                
            output_path = os.path.join(output_dir, f'similarity_{frame_file}')
            cv2.imwrite(output_path, combined)
            debug_time = time.time() - debug_start
        
        frame_time = time.time() - start_time
        total_time += frame_time
        frame_count += 1
    
    if frame_count > 0:
        avg_total_time = round(total_time / frame_count, 3)
        avg_model_time = round(model_time / frame_count, 3)
        print(f'Average total processing time per frame: {avg_total_time} seconds')
        print(f'Average model inference time per frame: {avg_model_time} seconds')
        print(f'Average times per frame:')
        print(f'  - Model inference: {round(model_time/frame_count, 3)} seconds') 
        print(f'  - Post-processing: {round(post_time, 3)} seconds')
        print(f'  - Source image loading: {round(source_load_time, 3)} seconds')
        print(f'  - Preprocessing: {round(preprocess_time, 3)} seconds')
        print(f'  - Compression: {round(compress_time, 3)} seconds')
        print(f'  - Saving: {round(save_time, 3)} seconds')
        if debug_mode:
            print(f'  - Debug visualization: {round(debug_time, 3)} seconds')
        print(f'Total frames processed: {frame_count}')
        print(f'Total time: {round(total_time, 1)} seconds')

def main():
    parser = argparse.ArgumentParser(description='Process frames and generate embeddings')
    parser.add_argument(
        '--frames_dir',
        type=str,
        default="path",
        help='Directory containing input frames'
    )
    parser.add_argument(
        '--output_dir', 
        type=str,
        default="path",
        help='Directory to save output features'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default="./openseg_model",
        help='Path to the model directory'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='model_replica_room0_checkpoints_pixel_emb10.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=10,
        help='Dimension of embeddings'
    )
    parser.add_argument(
        '--debug_mode',
        action='store_true',
        help='Save decoded and original heatmaps (will slow down processing)'
    )
    parser.add_argument(
        '--debug_prompt',
        type=str,
        default="lamp",
        help='Prompt for debug heatmaps'
    )
    
    args = parser.parse_args()
        
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    
    print(f'Processing frames from {args.frames_dir}')
    print(f'Saving outputs to {args.output_dir}')
    
    process_frames(args.frames_dir, args.embedding_dim, args.output_dir, args.model_path, args.checkpoint_path, args.debug_prompt, args.debug_mode)
    print('Processing complete!')

if __name__ == "__main__":
    main()
