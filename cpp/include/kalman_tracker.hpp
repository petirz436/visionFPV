#ifndef KALMAN_TRACKER_HPP
#define KALMAN_TRACKER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

/**
 * @brief 2D Kalman Filter Tracker for FPV Target Motion Prediction.
 * State Vector      : [x, y, vx, vy]^T
 * Measurement Vector: [zx, zy]^T
 */
class Kalman2DTracker {
public:
    Kalman2DTracker(float process_noise_std = 1e-2f, float measurement_noise_std = 1e-1f);
    ~Kalman2DTracker() = default;

    void init(float cx, float cy);
    void predict(float dt, float& pred_cx, float& pred_cy, float& vx, float& vy);
    void correct(float cx, float cy, float& est_cx, float& est_cy, float& vx, float& vy);

    bool isInitialized() const { return initialized_; }

private:
    cv::KalmanFilter kf_;
    bool initialized_{false};
};

#endif // KALMAN_TRACKER_HPP
