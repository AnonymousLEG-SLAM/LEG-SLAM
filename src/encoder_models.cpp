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

#include "include/encoder_models.h"


using namespace lf_encoder;


DinoV2Encoder::DinoV2Encoder(int input_width, int input_height, int output_pixels_count, int output_feat_dim, torch::DeviceType device_type) :
    LFEncoder(input_width, input_height, output_pixels_count, output_feat_dim) {
    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        device_type_= torch::kCUDA;
        std::cerr << "[LanguageFeaturesEncoder] The GPU is used to operate the encoder." << std::endl;
    } else {
        device_type_= torch::kCPU;
        std::cerr << "[LanguageFeaturesEncoder] The CPU is used to operate the encoder." << std::endl;
    }
}

void DinoV2Encoder::loadModel(const std::string &path_to_model) {
    try {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DinoV2");
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
            throw std::runtime_error("The input node doesn't contain the \"input_image\" name. The model cannot be loaded correctly.");
        }

        bool is_corr_output = false;
        for (size_t idx = 0; idx < session_->GetOutputCount(); ++idx) {
            Ort::AllocatedStringPtr output_node_name = session_->GetOutputNameAllocated(idx, allocator);
            if (output_node_name_.compare(output_node_name.get()) == 0) {
                is_corr_output = true;
                break;
            }
        }
        if (!is_corr_output) {
            throw std::runtime_error("The output nodes don't contain the \"x_norm_patchtokens\" name. The model cannot be loaded correctly.");
        }

        model_enable_ = true;
    } catch (const std::exception& exc) {
        std::cerr << "[DinoV2] " << std::string(exc.what()) << std::endl;
        throw std::runtime_error("Create session failed.");
    }

}

cv::Mat DinoV2Encoder::predict(const cv::Mat &image) {
    if (!model_enable_) {
        throw std::runtime_error("The encoder is not initialized. Before calling predict, execute loadModel.");
    }

    cv::Mat input_img;
    image.convertTo(input_img, CV_32F, 1. / 255.);
    // Transform tensor
    cv::resize(input_img, input_img, cv::Size(input_width_, input_height_));
    tensor_utils::normalize(input_img, input_img, mean_, std_);
    cv::Mat input_tensor_cv = cv::dnn::blobFromImage(input_img);

    if (!input_tensor_cv.isContinuous()) {
        input_tensor_cv = input_tensor_cv.clone();
    }

    auto input_shape = std::array<int64_t, 4>{1, input_img.channels(), input_width_, input_height_};
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info_,
                                                        input_tensor_cv.ptr<float>(0),
                                                        input_tensor_cv.total(),
                                                        input_shape.data(),
                                                        input_shape.size());

    std::array<int64_t, 3> output_shape{1, output_pixels_count_, output_feat_dim_};
    cv::Mat output_tensor_cv{output_pixels_count_, output_feat_dim_, CV_32F, 0.};
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info_,
                                                               output_tensor_cv.ptr<float>(0),
                                                               output_tensor_cv.total(),
                                                               output_shape.data(),
                                                               output_shape.size());

    std::vector<const char *> input_names{input_node_name_.data()}, output_names{output_node_name_.data()};
    session_->Run(options_, input_names.data(), &input_tensor, input_names.size(),
                  output_names.data(), &output_tensor, output_names.size());

    // Normalize features
    for (int idx = 0; idx < output_tensor_cv.rows; ++idx) {
        cv::normalize(output_tensor_cv.row(idx), output_tensor_cv.row(idx), 1.);
    }

    return output_tensor_cv;
}
