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

#ifndef ENCODER_MODELS_H
#define ENCODER_MODELS_H

#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn.hpp>

#include <torch/torch.h>
#include <torch/script.h>

//#include <tensorflow/cc/client/client_session.h>
//#include <tensorflow/cc/ops/io_ops.h>
//#include <tensorflow/core/framework/device_factory.h>
//#include <tensorflow/core/framework/tensor.h>
//#include <tensorflow/core/util/port.h>
//#include <tensorflow_cpp/model.h>

#include <onnxruntime_cxx_api.h>

#include "include/tensor_utils.h"


namespace lf_encoder {
    class LFEncoder {
    public:
        explicit LFEncoder(int input_width, int input_height, int output_pixels_count, int output_feat_dim) :
                input_width_(input_width), input_height_(input_height),
                output_pixels_count_(output_pixels_count), output_feat_dim_(output_feat_dim) {}
        virtual ~LFEncoder() = default;
        virtual void loadModel(const std::string &path_to_model) = 0;
        virtual cv::Mat predict(const cv::Mat &image) = 0;

    protected:
        int input_width_;
        int input_height_;
        int output_pixels_count_;
        int output_feat_dim_;
    };


//    class OpensegEncoder : public LFEncoder {
//    private:
//        inline std::tuple<tensorflow::Tensor, tensorflow::Tensor> getInputEncoder(const std::string &image_path);
//
//    public:
//        explicit OpensegEncoder(int encoder_output_size, torch::DeviceType device);
//        void loadModel(const std::string &path_to_model) override;
//        cv::Mat predict(const cv::Mat &image) override;
//
//    private:
//        torch::DeviceType device_type_;
//        std::string device_list_;
//        tensorflow_cpp::Model model_{};
//        int embeddings_size_{};
//    };

    class DinoV2Encoder : public LFEncoder {
    public:
        explicit DinoV2Encoder(int input_width, int input_height, int output_pixels_count, int output_feat_dim, torch::DeviceType device_type);
        void loadModel(const std::string &path_to_model) override;
        cv::Mat predict(const cv::Mat &image) override;

    private:
        const cv::Vec3f mean_{0.485, 0.456, 0.406};
        const cv::Vec3f std_{0.229, 0.224, 0.225};

        torch::DeviceType device_type_;
        bool model_enable_ = false;
        Ort::Env env_;
        std::unique_ptr<Ort::Session> session_;
        Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::RunOptions options_ = Ort::RunOptions{nullptr};

        std::string_view input_node_name_ = "input_image";
        std::string_view output_node_name_ = "x_norm_patchtokens";
    };
}

#endif // ENCODER_MODELS_H