#ifndef CAMERA_STREAM_HPP
#define CAMERA_STREAM_HPP

#include <opencv2/opencv.hpp>
#include <string>

/**
 * @brief Open VideoCapture device or stream URL with auto-fallback handling.
 * @param source Camera index (e.g. "0") or video file/URL path.
 * @return cv::VideoCapture opened instance or empty.
 */
cv::VideoCapture get_camera_capture(const std::string& source = "0");

#endif // CAMERA_STREAM_HPP
