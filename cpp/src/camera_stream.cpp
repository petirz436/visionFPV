#include "camera_stream.hpp"
#include <iostream>
#include <cctype>

cv::VideoCapture get_camera_capture(const std::string& source) {
    bool is_digit = true;
    for (char c : source) {
        if (!std::isdigit(c) && c != '-') {
            is_digit = false;
            break;
        }
    }

    if (is_digit) {
        int idx = std::stoi(source);
        int backends[] = { cv::CAP_V4L2, cv::CAP_ANY };

        for (int backend : backends) {
            cv::VideoCapture cap(idx, backend);
            if (cap.isOpened()) {
                cv::Mat frame;
                if (cap.read(frame) && !frame.empty()) {
                    std::cout << "[INFO] Camera successfully opened at index " << idx 
                              << " (backend=" << backend << ")." << std::endl;
                    return cap;
                }
                cap.release();
            }
        }

        std::cout << "[WARNING] Camera index " << idx << " failed. Scanning fallbacks..." << std::endl;
        for (int fb_idx = 0; fb_idx < 4; ++fb_idx) {
            if (fb_idx == idx) continue;
            for (int backend : backends) {
                cv::VideoCapture cap(fb_idx, backend);
                if (cap.isOpened()) {
                    cv::Mat frame;
                    if (cap.read(frame) && !frame.empty()) {
                        std::cout << "[INFO] Camera fallback success at index " << fb_idx << "." << std::endl;
                        return cap;
                    }
                    cap.release();
                }
            }
        }
    } else {
        cv::VideoCapture cap(source);
        if (cap.isOpened()) {
            std::cout << "[INFO] Stream/Video opened: " << source << std::endl;
            return cap;
        }
    }

    return cv::VideoCapture();
}
