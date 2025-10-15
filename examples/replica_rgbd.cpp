/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <memory>

#include <opencv2/core/core.hpp>

#include "ORB-SLAM3/include/System.h"
#include "include/language_features_encoder.h"
#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

void LoadImagesReplica(const std::filesystem::path &pathDir, std::vector <std::string> &vstrImageFilenamesRGB,
                std::vector <std::string> &vstrImageFilenamesD);

void LoadImagesScanNet(const std::filesystem::path &pathDir, std::vector<std::string> &vstrImageFilenamesRGB,
                       std::vector<std::string> &vstrImageFilenamesD);

void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath);

void saveGpuPeakMemoryUsage(std::filesystem::path pathSave);

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7 && argc != 8) {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_vocabulary"                        /*1*/
                  << " path_to_ORB_SLAM3_settings"                /*2*/
                  << " path_to_language_features_encoder_setting" /*3*/
                  << " path_to_gaussian_mapping_settings"         /*4*/
                  << " path_to_sequence"                          /*5*/
                  << " path_to_trajectory_output_directory/"      /*6*/
                  << " (optional)no_viewer"                       /*7*/
                  << std::endl;
        return 1;
    }
    bool use_viewer = true;
    if (argc == 8)
        use_viewer = (std::string(argv[7]) == "no_viewer" ? false : true);

    std::string output_directory = std::string(argv[6]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

    // Retrieve paths to images
    std::vector <std::string> vstrImageFilenamesRGB;
    std::vector <std::string> vstrImageFilenamesD;
    std::string strImageDir = std::string(argv[5]);
    std::filesystem::path pathImageDir(strImageDir);
    if (pathImageDir.string().find("Replica") != std::string::npos || pathImageDir.string().find("replica") != std::string::npos) {
        LoadImagesReplica(pathImageDir, vstrImageFilenamesRGB, vstrImageFilenamesD);
    } else if (pathImageDir.string().find("ScanNet") != std::string::npos || pathImageDir.string().find("scannet") != std::string::npos) {
        LoadImagesScanNet(pathImageDir, vstrImageFilenamesRGB, vstrImageFilenamesD);
    } else {
        std::cerr << std::endl << "Invalid dataset path." << std::endl;
        return 1;
    }

    // Check consistency in the number of images
    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty()) {
        std::cerr << std::endl << "No images found in provided path." << std::endl;
        return 1;
    } else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size()) {
        std::cerr << std::endl << "Different number of images for rgb and depth." << std::endl;
        return 1;
    }

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::shared_ptr <ORB_SLAM3::System> pSLAM =
            std::make_shared<ORB_SLAM3::System>(
                    argv[1], argv[2], ORB_SLAM3::System::RGBD);
    float imageScale = pSLAM->GetImageScale();

    // Create LanguageFeaturesEncoder
    std::filesystem::path lf_encoder_cfg_path(argv[3]);
    std::shared_ptr<lf_encoder::LanguageFeaturesEncoder> pLFEncoder =
            std::make_shared<lf_encoder::LanguageFeaturesEncoder>(
                    lf_encoder_cfg_path, device_type);

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(argv[4]);
    std::shared_ptr <GaussianMapper> pGausMapper =
            std::make_shared<GaussianMapper>(
                    pSLAM, gaussian_cfg_path, output_dir, 0, device_type);
    std::thread training_thd(&GaussianMapper::run, pGausMapper.get());

    // Create Gaussian Viewer
    std::thread viewer_thd;
    std::shared_ptr <ImGuiViewer> pViewer;
    if (use_viewer) {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << std::endl << std::endl;

    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();

    const int imageWidth = pSLAM->getSettings()->newImSize().width;
    const int imageHeight = pSLAM->getSettings()->newImSize().height;
    // Main loop
    cv::Mat imRGB, imD, imLF;
    for (int ni = 0; ni < nImages; ni++) {
        if (ni % 100 == 0) {
            std::cout << "Processing image: " << ni + 1 << " / " << nImages << std::endl;
        }
        if (pSLAM->isShutDown())
            break;
        // Read image and depthmap from file
        imRGB = cv::imread(vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        cv::cvtColor(imRGB, imRGB, CV_BGR2RGB);
        imD = cv::imread(vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        if (imRGB.size() != cv::Size(imageWidth, imageHeight)) {
            cv::resize(imRGB, imRGB, cv::Size(imageWidth, imageHeight));
        }
        if (imD.size() != cv::Size(imageWidth, imageHeight)) {
            cv::resize(imD, imD, cv::Size(imageWidth, imageHeight));
        }
        double tframe = ni;

        if (imRGB.empty()) {
            std::cerr << std::endl << "Failed to load image at: "
                      << vstrImageFilenamesRGB[ni] << std::endl;
            return 1;
        }
        if (imD.empty()) {
            std::cerr << std::endl << "Failed to load image at: "
                      << vstrImageFilenamesD[ni] << std::endl;
            return 1;
        }

        if (imageScale != 1.f) {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        imLF = pLFEncoder->createLanguageFeatures(imRGB);

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        pSLAM->TrackRGBDLF(imRGB, imD, imLF, tframe, std::vector<ORB_SLAM3::IMU::Point>(), vstrImageFilenamesRGB[ni]);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double >> (t2 - t1).count();
        vTimesTrack[ni] = ttrack;
    }

    std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
    double t_total = std::chrono::duration_cast<std::chrono::duration<double >> (t_end - t_start).count();
    std::cout << "Total time: " << t_total << " seconds" << std::endl;
    std::cout << "Average time per image: " << std::round(t_total / nImages * 1000) / 1000 << " milliseconds" << std::endl;
    std::cout << "Average FPS: " << std::round((nImages / t_total) * 10) / 10 << std::endl;

    // Stop all threads
    pSLAM->Shutdown();
    training_thd.join();
    if (use_viewer)
        viewer_thd.join();

    // GPU peak usage
    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");

    // Tracking time statistics
    saveTrackingTime(vTimesTrack, (output_dir / "TrackingTime.txt").string());

    // Save camera trajectory
    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
    pSLAM->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());

    return 0;
}

void LoadImagesReplica(const std::filesystem::path &pathDir, std::vector <std::string> &vstrImageFilenamesRGB,
                std::vector <std::string> &vstrImageFilenamesD) {
    std::filesystem::path pathImageDir = pathDir / "results";
    for (const auto &imagePath: std::filesystem::directory_iterator(pathImageDir)) {
        std::string name = imagePath.path().filename().string();
        if (name.rfind("frame", 0) == 0)
            vstrImageFilenamesRGB.push_back(imagePath.path().string());
        else if (name.rfind("depth", 0) == 0)
            vstrImageFilenamesD.push_back(imagePath.path().string());
        std::sort(vstrImageFilenamesRGB.begin(), vstrImageFilenamesRGB.end());
        std::sort(vstrImageFilenamesD.begin(), vstrImageFilenamesD.end());
    }
}

void LoadImagesScanNet(const std::filesystem::path &pathDir, std::vector<std::string> &vstrImageFilenamesRGB,
                       std::vector<std::string> &vstrImageFilenamesD) {
    std::filesystem::path colorDir = pathDir / "color";
    std::filesystem::path depthDir = pathDir / "depth";

    for (const auto &imagePath : std::filesystem::directory_iterator(colorDir)) {
        vstrImageFilenamesRGB.push_back(imagePath.path().string());
    }

    for (const auto &imagePath : std::filesystem::directory_iterator(depthDir)) {
        vstrImageFilenamesD.push_back(imagePath.path().string());
    }
    auto sortByValue = [](const std::string &a, const std::string &b) {
        int numA = std::stoi(a.substr(a.find_last_of('/') + 1));
        int numB = std::stoi(b.substr(b.find_last_of('/') + 1));
        return numA < numB;
    };

    std::sort(vstrImageFilenamesRGB.begin(), vstrImageFilenamesRGB.end(), sortByValue);
    std::sort(vstrImageFilenamesD.begin(), vstrImageFilenamesD.end(), sortByValue);
}

void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath) {
    std::ofstream out;
    out.open(strSavePath.c_str());
    std::size_t nImages = vTimesTrack.size();
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++) {
        out << std::fixed << std::setprecision(4)
            << vTimesTrack[ni] << std::endl;
        totaltime += vTimesTrack[ni];
    }

    // std::sort(vTimesTrack.begin(), vTimesTrack.end());
    // out << "-------" << std::endl;
    // out << std::fixed << std::setprecision(4)
    //     << "median tracking time: " << vTimesTrack[nImages / 2] << std::endl;
    // out << std::fixed << std::setprecision(4)
    //     << "mean tracking time: " << totaltime / nImages << std::endl;

    out.close();
}

void saveGpuPeakMemoryUsage(std::filesystem::path pathSave) {
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    c10Alloc::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_reserved_MB = reserved_bytes.peak / (1024.0 * 1024.0);

    c10Alloc::Stat alloc_bytes = mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_alloc_MB = alloc_bytes.peak / (1024.0 * 1024.0);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << std::endl;
    out << "Peak allocated (MB): " << max_alloc_MB << std::endl;
    out.close();
}