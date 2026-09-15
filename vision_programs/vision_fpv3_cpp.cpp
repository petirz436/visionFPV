#include "vision_fpv3.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

std::pair<cv::Ptr<cv::Tracker>, std::string> create_tracker(
    const std::string& tracker_type,
    const std::string& models_dir
) {
    std::string type_lower = tracker_type;
    std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(), ::tolower);

    // High Speed CSRT Tracker
    if (type_lower == "csrt" || type_lower == "siamrpn") {
        try {
            cv::Ptr<cv::TrackerCSRT> tracker = cv::TrackerCSRT::create();
            if (tracker) {
                return { tracker, "csrt" };
            }
        } catch (const std::exception& e) {
            std::cout << "[WARNING] CSRT Tracker creation failed: " << e.what() << std::endl;
        }
    }

    // High Speed KCF Tracker fallback
    try {
        cv::Ptr<cv::TrackerKCF> tracker = cv::TrackerKCF::create();
        if (tracker) {
            return { tracker, "kcf" };
        }
    } catch (...) {}

    throw std::runtime_error("No supported Tracker engine found in OpenCV installation.");
}

cv::Rect2d clamp_bbox(const cv::Rect2d& bbox, int frame_width, int frame_height) {
    double x = std::max(0.0, std::min(bbox.x, static_cast<double>(frame_width - 1)));
    double y = std::max(0.0, std::min(bbox.y, static_cast<double>(frame_height - 1)));
    double w = std::max(10.0, std::min(bbox.width, static_cast<double>(frame_width) - x));
    double h = std::max(10.0, std::min(bbox.height, static_cast<double>(frame_height) - y));
    return cv::Rect2d(x, y, w, h);
}

void calculate_error(float target_cx, float target_cy, int frame_width, int frame_height, float& err_x, float& err_y) {
    float frame_cx = frame_width / 2.0f;
    float frame_cy = frame_height / 2.0f;
    err_x = target_cx - frame_cx;
    err_y = target_cy - frame_cy;
}

float apply_deadzone(float value, float deadzone) {
    if (std::abs(value) < deadzone) {
        return 0.0f;
    }
    return value;
}
