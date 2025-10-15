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

#include "include/language_features_encoder.h"

using namespace lf_encoder;


std::tuple<LanguageFeaturesEncoder::EncoderParams, LanguageFeaturesEncoder::CompressorParams>
LanguageFeaturesEncoder::readConfigFromFile(const std::filesystem::path &config_file_path) {
    cv::FileStorage settings_file(config_file_path.string(), cv::FileStorage::READ);
    if (!settings_file.isOpened()) {
        std::stringstream message_stream;
        message_stream << "Failed to open settings file at: " << config_file_path;
        throw std::runtime_error(message_stream.str());
    }

    std::cerr << "[LanguageFeaturesEncoder] Reading parameters from " << config_file_path << std::endl;

    auto encoder_type = settings_file["Encoder.Type"].operator std::string();
    auto path_to_encoder_model = settings_file["Encoder.Path"].operator std::string();
    auto encoder_input_width = settings_file["Encoder.InputImage.Width"].operator int();
    auto encoder_input_height = settings_file["Encoder.InputImage.Height"].operator int();
    auto encoder_output_pixels = settings_file["Encoder.OutputFeatures.Pixels"].operator int();
    auto encoder_output_feat_dim = settings_file["Encoder.OutputFeatures.EmbeddingSize"].operator int();

    auto compressor_type = settings_file["PixelwiseCompressor.Type"].operator std::string();
    auto path_to_compressor_model = settings_file["PixelwiseCompressor.Path"].operator std::string();
    auto compress_embeddings_size = settings_file["PixelwiseCompressor.CompressedEmbeddingSize"].operator int();

    return {{encoder_type, path_to_encoder_model, encoder_input_width, encoder_input_height, encoder_output_pixels, encoder_output_feat_dim},
            {compressor_type, path_to_compressor_model, encoder_output_pixels, encoder_output_feat_dim, compress_embeddings_size}};
}

LanguageFeaturesEncoder::LanguageFeaturesEncoder(std::filesystem::path encoder_config_file_path, torch::DeviceType device_type) :
        encoder_config_file_path_(encoder_config_file_path) {
//    device_type_ = torch::cuda::is_available() ? device_type : torch::kCPU;
    if (torch::cuda::is_available() && device_type == torch::kCUDA) {  // TODO: It is necessary to provide automatic selection of the memory size.
        c10::cuda::CUDACachingAllocator::init(1);
        c10::cuda::CUDACachingAllocator::setMemoryFraction(0.8, 0);
        device_type_ = torch::kCUDA;
    } else {
        device_type_ = torch::kCPU;
    }

    auto [encoder_config, compressor_config] = readConfigFromFile(encoder_config_file_path_);
    if (encoder_config.type == "DinoV2") {
        encoder_ = std::make_unique<DinoV2Encoder>(encoder_config.input_width, encoder_config.input_height,
                                                   encoder_config.output_pixels_count, encoder_config.output_feat_dim,
                                                   device_type_);
    } else if (encoder_config.type == "OpenSeg") {
        throw std::runtime_error("No implementation");
    } else {
        throw std::runtime_error("Invalid encoder type");
    }
    encoder_->loadModel(encoder_config.path);

    if (compressor_config.type == "PCA") {
        compressor_ = std::make_unique<PCA>(compressor_config.pixels_count, compressor_config.input_feat_dim,
                                            compressor_config.output_feat_dim, device_type_);
    } else if (compressor_config.type == "MLP") {
        throw std::runtime_error("No implementation");
    } else {
        throw std::runtime_error("Invalid compressor type");
    }
    compressor_->loadModel(compressor_config.path);


}

cv::Mat LanguageFeaturesEncoder::createLanguageFeatures(const cv::Mat &rgb_image) {
    auto language_features = encoder_->predict(rgb_image);
    auto compressed_language_features = compressor_->predict(language_features);
    int language_features_image_size = static_cast<int>(std::sqrt(compressed_language_features.rows));
    auto language_features_image = compressed_language_features.reshape(compressed_language_features.cols, language_features_image_size);
    return language_features_image;
}
