#ifndef SIAMRPN_TRACKER_HPP
#define SIAMRPN_TRACKER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <Python.h>

/**
 * @brief SiamRPN (DaSiamRPN ONNX) Target Tracker Engine for C++.
 * Integrates high-performance OpenCV DaSiamRPN engine for exact pixel accuracy.
 */
class SiamRPNTracker {
public:
    SiamRPNTracker();
    ~SiamRPNTracker();

    bool load_models(const std::string& models_dir = "models");
    bool init(const cv::Mat& frame, const cv::Rect2d& bbox);
    bool update(const cv::Mat& frame, cv::Rect2d& bbox);

    bool is_initialized() const { return initialized_; }

private:
    bool models_loaded_{false};
    bool initialized_{false};

    PyObject* pTracker_{nullptr};
    PyObject* pModule_{nullptr};
    PyObject* pDict_{nullptr};
};

#endif // SIAMRPN_TRACKER_HPP
