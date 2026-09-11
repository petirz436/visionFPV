#include "kalman_tracker.hpp"

Kalman2DTracker::Kalman2DTracker(float process_noise_std, float measurement_noise_std)
    : kf_(4, 2, 0, CV_32F) {

    // State transition matrix F (constant velocity model)
    kf_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);

    // Measurement matrix H
    kf_.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    // Process noise covariance Q
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(process_noise_std));
    kf_.processNoiseCov.at<float>(2, 2) *= 5.0f; // Allow high acceleration
    kf_.processNoiseCov.at<float>(3, 3) *= 5.0f;

    // Measurement noise covariance R
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(measurement_noise_std));

    // Error covariance P
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0f));

    initialized_ = false;
}

void Kalman2DTracker::init(float cx, float cy) {
    kf_.statePost.at<float>(0) = cx;
    kf_.statePost.at<float>(1) = cy;
    kf_.statePost.at<float>(2) = 0.0f;
    kf_.statePost.at<float>(3) = 0.0f;

    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0f));
    initialized_ = true;
}

void Kalman2DTracker::predict(float dt, float& pred_cx, float& pred_cy, float& vx, float& vy) {
    if (!initialized_) {
        pred_cx = pred_cy = vx = vy = 0.0f;
        return;
    }

    kf_.transitionMatrix.at<float>(0, 2) = dt;
    kf_.transitionMatrix.at<float>(1, 3) = dt;

    cv::Mat prediction = kf_.predict();
    pred_cx = prediction.at<float>(0);
    pred_cy = prediction.at<float>(1);
    vx = prediction.at<float>(2);
    vy = prediction.at<float>(3);
}

void Kalman2DTracker::correct(float cx, float cy, float& est_cx, float& est_cy, float& vx, float& vy) {
    if (!initialized_) {
        init(cx, cy);
        est_cx = cx;
        est_cy = cy;
        vx = 0.0f;
        vy = 0.0f;
        return;
    }

    cv::Mat measurement = (cv::Mat_<float>(2, 1) << cx, cy);
    cv::Mat estimated = kf_.correct(measurement);

    est_cx = estimated.at<float>(0);
    est_cy = estimated.at<float>(1);
    vx = estimated.at<float>(2);
    vy = estimated.at<float>(3);
}
