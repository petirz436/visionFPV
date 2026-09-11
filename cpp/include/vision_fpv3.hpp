#ifndef VISION_FPV3_HPP
#define VISION_FPV3_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <memory>
#include <utility>

#include "siamrpn_tracker.hpp"

// Configuration & Hyperparameters
constexpr int DEFAULT_ROI_W = 100;
constexpr int DEFAULT_ROI_H = 100;
constexpr float ALPHA = 0.35f;       // EMA smoothing factor
constexpr float DEADZONE = 20.0f;    // Pixel deadzone
constexpr int MAX_LOST_FRAMES = 5;

/**
 * @brief Clamp Bounding Box to image dimensions.
 */
cv::Rect2d clamp_bbox(const cv::Rect2d& bbox, int frame_width, int frame_height);

/**
 * @brief Calculate position error relative to camera center.
 */
void calculate_error(float target_cx, float target_cy, int frame_width, int frame_height, float& err_x, float& err_y);

/**
 * @brief Apply pixel deadzone threshold.
 */
float apply_deadzone(float value, float deadzone = DEADZONE);

#endif // VISION_FPV3_HPP
