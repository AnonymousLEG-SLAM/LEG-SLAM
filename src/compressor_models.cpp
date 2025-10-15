/**
 * This file is part of LEGS-SLAM
 *
 * LEGS-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * LEGS-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with LEGS-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "include/compressor_models.h"

using namespace lf_encoder;


PCA::PCA(int pixels_count, int input_feat_dim, int output_feat_dim, torch::DeviceType device_type) :
        PixelwiseCompressor(pixels_count, input_feat_dim, output_feat_dim) {
    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        device_type_= torch::kCUDA;
        std::cerr << "[PixelwiseCompressor] The GPU is used to operate the compressor." << std::endl;
    } else {
        device_type_= torch::kCPU;
        std::cerr << "[PixelwiseCompressor] The CPU is used to operate the compressor." << std::endl;
    }
}

void PCA::loadModel(const std::string &path_to_model) {
    try {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "PCA");
        Ort::SessionOptions session_option;
        if (device_type_ == torch::kCUDA) {
            OrtCUDAProviderOptions cuda_option;
            cuda_option.device_id = 0;  // TODO: Query device id
            session_option.AppendExecutionProvider_CUDA(cuda_option);
        }

        session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, path_to_model.c_str(), session_option);
        if (session_->GetInputCount() != 1) {
            throw std::runtime_error("Unacceptable number of inputs to the model. You must have exactly 1 input.");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr input_node_name = session_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions{});
        if (input_node_name_.compare(input_node_name.get()) != 0) {
            throw std::runtime_error("The input node doesn't contain the \"input_feat\" name. The model cannot be loaded correctly.");
        }

        if (session_->GetOutputCount() != 1) {
            throw std::runtime_error("Unacceptable number of outputs to the model. You must have exactly 1 output.");
        }
        Ort::AllocatedStringPtr output_node_name = session_->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions{});
        if (output_node_name_.compare(output_node_name.get()) != 0) {
            throw std::runtime_error("The output node doesn't contain the \"compressed_feat\" name. The model cannot be loaded correctly.");
        }

        model_enable_ = true;
    } catch (const std::exception& exc) {
        std::cerr << "[PCA] " << std::string(exc.what()) << std::endl;
        throw std::runtime_error("Create session failed.");
    }
}

cv::Mat PCA::predict(cv::Mat input_language_features) {
    if (!model_enable_) {
        throw std::runtime_error("The compressor is not initialized. Before calling predict, execute loadModel.");
    }

    if (!input_language_features.isContinuous()) {
        input_language_features = input_language_features.clone();
    }

    std::array<int64_t, 2> input_shape{pixels_count_, input_feat_dim_};
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info_,
                                                               input_language_features.ptr<float>(0),
                                                               input_language_features.total(),
                                                               input_shape.data(),
                                                               input_shape.size());

    std::array<int64_t, 2> output_shape{pixels_count_, output_feat_dim_};
    cv::Mat compressed_language_features{pixels_count_, output_feat_dim_, CV_32F, 0.0};
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info_,
                                                               compressed_language_features.ptr<float>(0),
                                                               compressed_language_features.total(),
                                                               output_shape.data(),
                                                               output_shape.size());

    std::vector<const char *> input_names{input_node_name_.data()}, output_names{output_node_name_.data()};
    session_->Run(options_, input_names.data(), &input_tensor, input_names.size(),
                  output_names.data(), &output_tensor, output_names.size());

    return compressed_language_features;
}
