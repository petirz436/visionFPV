#include "camera_stream.hpp"
#include "kalman_tracker.hpp"
#include "siamrpn_tracker.hpp"
#include "vision_fpv3.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <string>

int main(int argc, char** argv) {
    std::string video_path = "../data/synthetic_fpv_test.mp4";
    std::string out_json_path = "../data/cpp_eval_results.json";

    if (argc > 1) video_path = argv[1];
    if (argc > 2) out_json_path = argv[2];

    if (!std::ifstream(video_path).good()) {
        if (std::ifstream("data/synthetic_fpv_test.mp4").good()) {
            video_path = "data/synthetic_fpv_test.mp4";
        }
    }

    // Frame 0 Ground Truth ROI (x=50, y=120, w=60, h=60)
    cv::Rect2d init_bbox(50, 120, 60, 60);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Could not open video: " << video_path << std::endl;
        return -1;
    }

    cv::Mat first_frame;
    if (!cap.read(first_frame) || first_frame.empty()) {
        std::cerr << "[ERROR] Could not read first frame." << std::endl;
        return -1;
    }

    int frame_width = first_frame.cols;
    int frame_height = first_frame.rows;

    SiamRPNTracker tracker;
    std::string models_dir = "models";
    if (!std::ifstream("models/dasiamrpn_model.onnx").good() && std::ifstream("../models/dasiamrpn_model.onnx").good()) {
        models_dir = "../models";
    }

    if (!tracker.load_models(models_dir)) {
        std::cerr << "[ERROR] Could not load SiamRPN ONNX models." << std::endl;
        return -1;
    }

    tracker.init(first_frame, init_bbox);

    Kalman2DTracker kalman;
    kalman.init(static_cast<float>(init_bbox.x + init_bbox.width / 2.0),
                static_cast<float>(init_bbox.y + init_bbox.height / 2.0));

    bool tracking = true;
    int lost_counter = 0;
    cv::Size2d last_bbox_size = init_bbox.size();

    std::ofstream out_f(out_json_path);
    if (!out_f.is_open()) {
        std::cerr << "[ERROR] Could not open output JSON: " << out_json_path << std::endl;
        return -1;
    }

    out_f << "[\n";

    // Frame 0 initial record
    out_f << "  {\n"
          << "    \"frame\": 0,\n"
          << "    \"x\": " << init_bbox.x << ",\n"
          << "    \"y\": " << init_bbox.y << ",\n"
          << "    \"w\": " << init_bbox.width << ",\n"
          << "    \"h\": " << init_bbox.height << ",\n"
          << "    \"ms\": 0.1,\n"
          << "    \"status\": \"TRACKED\"\n"
          << "  }";

    int frame_idx = 0;
    cv::Mat frame;

    while (cap.read(frame) && !frame.empty()) {
        frame_idx++;

        auto t0 = std::chrono::high_resolution_clock::now();
        cv::Rect2d bbox(0, 0, 0, 0);
        std::string status = "LOST";

        if (tracking) {
            bool success = tracker.update(frame, bbox);
            if (success) {
                bool valid = (bbox.width > 5 && bbox.height > 5 &&
                              bbox.x + bbox.width > 0 && bbox.y + bbox.height > 0 &&
                              bbox.x < frame_width && bbox.y < frame_height);
                if (valid) {
                    float meas_cx = static_cast<float>(bbox.x + bbox.width / 2.0);
                    float meas_cy = static_cast<float>(bbox.y + bbox.height / 2.0);
                    last_bbox_size = bbox.size();

                    float est_cx, est_cy, vx, vy;
                    kalman.correct(meas_cx, meas_cy, est_cx, est_cy, vx, vy);
                    
                    float pred_cx, pred_cy, pvx, pvy;
                    kalman.predict(1.0f, pred_cx, pred_cy, pvx, pvy);

                    status = "TRACKED";
                    lost_counter = 0;
                } else {
                    success = false;
                }
            }

            if (!success) {
                lost_counter++;
                if (lost_counter <= MAX_LOST_FRAMES) {
                    float pred_cx, pred_cy, vx, vy;
                    kalman.predict(1.0f, pred_cx, pred_cy, vx, vy);

                    double pred_x = static_cast<double>(pred_cx - last_bbox_size.width / 2.0);
                    double pred_y = static_cast<double>(pred_cy - last_bbox_size.height / 2.0);
                    bbox = clamp_bbox(cv::Rect2d(pred_x, pred_y, last_bbox_size.width, last_bbox_size.height), frame_width, frame_height);

                    status = "PREDICTING";
                    try {
                        tracker.init(frame, bbox);
                    } catch (...) {}
                } else {
                    tracking = false;
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        out_f << ",\n  {\n"
              << "    \"frame\": " << frame_idx << ",\n"
              << "    \"x\": " << bbox.x << ",\n"
              << "    \"y\": " << bbox.y << ",\n"
              << "    \"w\": " << bbox.width << ",\n"
              << "    \"h\": " << bbox.height << ",\n"
              << "    \"ms\": " << elapsed_ms << ",\n"
              << "    \"status\": \"" << status << "\"\n"
              << "  }";
    }

    out_f << "\n]\n";
    out_f.close();

    std::cout << "[BENCHMARK C++] Processed " << (frame_idx + 1) << " frames. Saved to '" << out_json_path << "'." << std::endl;
    return 0;
}
