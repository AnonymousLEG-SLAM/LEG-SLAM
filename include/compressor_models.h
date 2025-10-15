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

#ifndef COMPRESSOR_MODELS_H
#define COMPRESSOR_MODELS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <torch/torch.h>
#include <onnxruntime_cxx_api.h>

#include "include/tensor_utils.h"

namespace lf_encoder {
    class PixelwiseCompressor {
    public:
        explicit PixelwiseCompressor(int pixels_count, int input_feat_dim, int output_feat_dim) :
                pixels_count_(pixels_count), input_feat_dim_(input_feat_dim), output_feat_dim_(output_feat_dim) {}
        virtual ~PixelwiseCompressor() = default;
        virtual void loadModel(const std::string &path_to_model) = 0;
        virtual cv::Mat predict(cv::Mat language_features) = 0;
    protected:
        int pixels_count_, input_feat_dim_, output_feat_dim_;
    };


    class PCA : public PixelwiseCompressor {
    public:
        explicit PCA(int pixels_count, int input_feat_dim, int output_feat_dim, torch::DeviceType device_type);
        void loadModel(const std::string &path_to_model) override;
        cv::Mat predict(cv::Mat language_features) override;

    private:
        torch::DeviceType device_type_;

        bool model_enable_ = false;
        Ort::Env env_;
        std::unique_ptr<Ort::Session> session_;
        Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::RunOptions options_ = Ort::RunOptions{nullptr};

        std::string_view input_node_name_ = "input_feat";
        std::string_view output_node_name_ = "variable";
    };
}

#endif // COMPRESSOR_MODELS_H
