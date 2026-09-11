#include "camera_stream.hpp"
#include "kalman_tracker.hpp"
#include "siamrpn_tracker.hpp"
#include "vision_fpv3.hpp"

#include <iostream>
#include <chrono>
#include <iomanip>

// Global mouse interaction state
bool selecting = false;
bool dragging = false;
cv::Point start_point;
cv::Point current_point;
cv::Rect2d selected_bbox;
bool has_selected_bbox = false;

static void mouse_callback(int event, int x, int y, int flags, void* param) {
    (void)flags;
    (void)param;

    if (event == cv::EVENT_LBUTTONDOWN) {
        selecting = true;
        dragging = false;
        start_point = cv::Point(x, y);
        current_point = cv::Point(x, y);
    } else if (event == cv::EVENT_MOUSEMOVE) {
        if (selecting) {
            current_point = cv::Point(x, y);
            if (std::abs(current_point.x - start_point.x) > 5 || std::abs(current_point.y - start_point.y) > 5) {
                dragging = true;
            }
        }
    } else if (event == cv::EVENT_LBUTTONUP) {
        if (!selecting) return;
        selecting = false;

        int x1 = std::min(start_point.x, x);
        int y1 = std::min(start_point.y, y);
        int x2 = std::max(start_point.x, x);
        int y2 = std::max(start_point.y, y);

        int w = x2 - x1;
        int h = y2 - y1;

        if (w < 10 || h < 10) {
            x1 = x - DEFAULT_ROI_W / 2;
            y1 = y - DEFAULT_ROI_H / 2;
            w = DEFAULT_ROI_W;
            h = DEFAULT_ROI_H;
        }

        selected_bbox = cv::Rect2d(x1, y1, w, h);
        has_selected_bbox = true;
    }
}

int main(int argc, char** argv) {
    std::string camera_src = (argc > 1) ? argv[1] : "0";

    cv::VideoCapture cap = get_camera_capture(camera_src);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Could not open video capture source." << std::endl;
        return -1;
    }

    std::string window_name = "FPV Target Tracker v3.0 [C++] - SiamRPN 2 + Kalman";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(window_name, mouse_callback, nullptr);

    // Trackers
    SiamRPNTracker siam_tracker;
    cv::Ptr<cv::Tracker> csrt_tracker;

    std::string active_tracker_name = "siamrpn";
    bool tracking = false;
    Kalman2DTracker kalman;

    int lost_counter = 0;
    bool is_predicting = false;

    float smooth_cx = -1.0f;
    float smooth_cy = -1.0f;
    cv::Size2d last_bbox_size(DEFAULT_ROI_W, DEFAULT_ROI_H);
    cv::Point2f target_velocity(0.0f, 0.0f);

    auto last_time = std::chrono::high_resolution_clock::now();

    std::cout << "\n==========================================================" << std::endl;
    std::cout << " FPV DRONE TARGET TRACKER v3.0.0 (C++ SiamRPN Edition)" << std::endl;
    std::cout << " Engine : SiamRPN (DaSiamRPN ONNX) / CSRT + 2D Kalman" << std::endl;
    std::cout << " Platform: Linux / ARM (Optimized NEON & TBB)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "[MOUSE]" << std::endl;
    std::cout << "  Drag  : Select ROI Bounding Box" << std::endl;
    std::cout << "  Click : Select point target (Default ROI)" << std::endl;
    std::cout << "[KEYBOARD]" << std::endl;
    std::cout << "  T     : Toggle Tracker Engine (SiamRPN <-> CSRT)" << std::endl;
    std::cout << "  R     : Reset Tracker" << std::endl;
    std::cout << "  ESC   : Quit" << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    bool is_video_file = (cap.get(cv::CAP_PROP_FRAME_COUNT) > 0);
    double cap_fps = cap.get(cv::CAP_PROP_FPS);
    int wait_delay = (is_video_file && cap_fps > 0) ? static_cast<int>(1000.0 / cap_fps) : 1;
    if (wait_delay < 1) wait_delay = 1;

    cv::Mat raw_frame;
    while (true) {
        if (!cap.read(raw_frame) || raw_frame.empty()) {
            if (is_video_file) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                if (!cap.read(raw_frame) || raw_frame.empty()) {
                    std::cout << "[INFO] End of stream or frame failed." << std::endl;
                    break;
                }
            } else {
                std::cout << "[INFO] End of stream or frame failed." << std::endl;
                break;
            }
        }

        int frame_width = raw_frame.cols;
        int frame_height = raw_frame.rows;
        int frame_cx = frame_width / 2;
        int frame_cy = frame_height / 2;

        cv::Mat display_frame = raw_frame.clone();

        // Draw camera center crosshair
        cv::drawMarker(display_frame, cv::Point(frame_cx, frame_cy), cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 20, 2);

        // 1. User ROI Dragging Preview
        if (selecting) {
            int x1 = std::min(start_point.x, current_point.x);
            int y1 = std::min(start_point.y, current_point.y);
            int x2 = std::max(start_point.x, current_point.x);
            int y2 = std::max(start_point.y, current_point.y);
            cv::rectangle(display_frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 255), 2);
        }

        // 2. Initialize Tracker with New ROI
        if (has_selected_bbox) {
            cv::Rect2d bbox = clamp_bbox(selected_bbox, frame_width, frame_height);
            try {
                if (active_tracker_name == "siamrpn") {
                    if (siam_tracker.init(raw_frame, bbox)) {
                        tracking = true;
                    } else {
                        std::cout << "[WARNING] SiamRPN init failed. Falling back to CSRT." << std::endl;
                        active_tracker_name = "csrt";
                        csrt_tracker = cv::TrackerCSRT::create();
                        csrt_tracker->init(raw_frame, bbox);
                        tracking = true;
                    }
                } else {
                    csrt_tracker = cv::TrackerCSRT::create();
                    csrt_tracker->init(raw_frame, bbox);
                    tracking = true;
                }

                float init_cx = static_cast<float>(bbox.x + bbox.width / 2.0);
                float init_cy = static_cast<float>(bbox.y + bbox.height / 2.0);

                kalman.init(init_cx, init_cy);
                smooth_cx = init_cx;
                smooth_cy = init_cy;
                last_bbox_size = bbox.size();
                lost_counter = 0;
                is_predicting = false;

                std::cout << "[TRACKER] Target locked (" << active_tracker_name << "): Center=(" 
                          << init_cx << ", " << init_cy << ")" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Tracker init failed: " << e.what() << std::endl;
                tracking = false;
            }
            has_selected_bbox = false;
        }

        // 3. Tracking & Motion Prediction
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        if (dt <= 0.0f) dt = 0.001f;
        last_time = now;

        float target_cx = -1.0f;
        float target_cy = -1.0f;
        bool has_target = false;

        if (tracking) {
            cv::Rect2d bbox;
            bool success = false;

            if (active_tracker_name == "siamrpn") {
                success = siam_tracker.update(raw_frame, bbox);
            } else if (csrt_tracker) {
                success = csrt_tracker->update(raw_frame, bbox);
            }

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

                    target_cx = est_cx;
                    target_cy = est_cy;
                    target_velocity = cv::Point2f(vx, vy);
                    has_target = true;

                    lost_counter = 0;
                    is_predicting = false;

                    cv::Scalar box_color = (active_tracker_name == "siamrpn") ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 255, 0);
                    cv::rectangle(display_frame, bbox, box_color, 2);
                } else {
                    success = false;
                }
            }

            if (!success) {
                lost_counter++;
                if (lost_counter <= MAX_LOST_FRAMES) {
                    float pred_cx, pred_cy, vx, vy;
                    kalman.predict(1.0f, pred_cx, pred_cy, vx, vy);
                    target_cx = pred_cx;
                    target_cy = pred_cy;
                    target_velocity = cv::Point2f(vx, vy);
                    has_target = true;
                    is_predicting = true;

                    double pred_x = static_cast<double>(pred_cx - last_bbox_size.width / 2.0);
                    double pred_y = static_cast<double>(pred_cy - last_bbox_size.height / 2.0);
                    cv::Rect2d pred_bbox = clamp_bbox(cv::Rect2d(pred_x, pred_y, last_bbox_size.width, last_bbox_size.height), frame_width, frame_height);

                    try {
                        if (active_tracker_name == "siamrpn") {
                            siam_tracker.init(raw_frame, pred_bbox);
                        } else if (csrt_tracker) {
                            csrt_tracker = cv::TrackerCSRT::create();
                            csrt_tracker->init(raw_frame, pred_bbox);
                        }
                    } catch (...) {}

                    cv::rectangle(display_frame, pred_bbox, cv::Scalar(0, 165, 255), 2);
                } else {
                    tracking = false;
                    is_predicting = false;
                }
            }
        }

        // 4. Smoothing & Controller Error Computation
        if (has_target) {
            if (smooth_cx < 0.0f) smooth_cx = target_cx;
            if (smooth_cy < 0.0f) smooth_cy = target_cy;

            smooth_cx = ALPHA * target_cx + (1.0f - ALPHA) * smooth_cx;
            smooth_cy = ALPHA * target_cy + (1.0f - ALPHA) * smooth_cy;

            float error_x, error_y;
            calculate_error(smooth_cx, smooth_cy, frame_width, frame_height, error_x, error_y);
            float err_x_ctl = apply_deadzone(error_x, DEADZONE);
            float err_y_ctl = apply_deadzone(error_y, DEADZONE);

            cv::Point target_center(static_cast<int>(smooth_cx), static_cast<int>(smooth_cy));
            cv::Scalar center_color = is_predicting ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 0, 255);
            cv::circle(display_frame, target_center, 6, center_color, -1);

            // Trajectory vector line
            cv::line(display_frame, cv::Point(frame_cx, frame_cy), target_center, cv::Scalar(255, 255, 255), 2);

            // Velocity arrow line
            if (std::abs(target_velocity.x) > 0.5f || std::abs(target_velocity.y) > 0.5f) {
                cv::Point vel_end(
                    static_cast<int>(smooth_cx + target_velocity.x * 5.0f),
                    static_cast<int>(smooth_cy + target_velocity.y * 5.0f)
                );
                cv::arrowedLine(display_frame, target_center, vel_end, cv::Scalar(0, 255, 255), 2, 8, 0, 0.3);
            }

            // HUD Telemetry Text (Identical to Python visionFPV3.py)
            std::string status_text = is_predicting ? 
                ("PREDICTING [KALMAN] (" + std::to_string(lost_counter) + "/" + std::to_string(MAX_LOST_FRAMES) + ")") :
                ("TRACKING (" + active_tracker_name + ")");
            cv::Scalar status_color = is_predicting ? cv::Scalar(0, 165, 255) : ((active_tracker_name == "siamrpn") ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 255, 0));

            cv::putText(display_frame, status_text, cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2);

            char buf[128];
            std::snprintf(buf, sizeof(buf), "CX: %.1f", smooth_cx);
            cv::putText(display_frame, buf, cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2);

            std::snprintf(buf, sizeof(buf), "CY: %.1f", smooth_cy);
            cv::putText(display_frame, buf, cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2);

            std::snprintf(buf, sizeof(buf), "ERR X: %.1f", err_x_ctl);
            cv::putText(display_frame, buf, cv::Point(20, 130), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2);

            std::snprintf(buf, sizeof(buf), "ERR Y: %.1f", err_y_ctl);
            cv::putText(display_frame, buf, cv::Point(20, 160), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2);

            float norm_x = std::max(-1.0f, std::min(1.0f, error_x / (frame_width / 2.0f)));
            float norm_y = std::max(-1.0f, std::min(1.0f, error_y / (frame_height / 2.0f)));
            std::snprintf(buf, sizeof(buf), "NORM X: %+.2f", norm_x);
            cv::putText(display_frame, buf, cv::Point(20, 195), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2);

            std::snprintf(buf, sizeof(buf), "NORM Y: %+.2f", norm_y);
            cv::putText(display_frame, buf, cv::Point(20, 225), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2);

            std::snprintf(buf, sizeof(buf), "VEL: (%+.1f, %+.1f)", target_velocity.x, target_velocity.y);
            cv::putText(display_frame, buf, cv::Point(20, 255), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);

        } else {
            std::string status_text = (!tracking && lost_counter > MAX_LOST_FRAMES) ? "TARGET LOST" : "CLICK / DRAG TARGET";
            cv::Scalar status_color = (status_text == "TARGET LOST") ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255);
            cv::putText(display_frame, status_text, cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2);
        }

        // 5. Deadzone & FPS Overlay
        int dz_val = static_cast<int>(DEADZONE);
        cv::rectangle(display_frame, cv::Point(frame_cx - dz_val, frame_cy - dz_val), cv::Point(frame_cx + dz_val, frame_cy + dz_val), cv::Scalar(255, 255, 255), 1);

        // Active Engine status
        cv::putText(display_frame, ("ENGINE: " + active_tracker_name), cv::Point(frame_width - 240, 65), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

        float fps = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
        char fps_buf[64];
        std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps);
        cv::putText(display_frame, fps_buf, cv::Point(frame_width - 150, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        cv::imshow(window_name, display_frame);

        int key = cv::waitKey(wait_delay) & 0xFF;
        if (key == 27) { // ESC
            break;
        } else if (key == 'r' || key == 'R') {
            std::cout << "\n[TRACKER] Tracker Reset." << std::endl;
            tracking = false;
            lost_counter = 0;
            is_predicting = false;
            smooth_cx = smooth_cy = -1.0f;
        } else if (key == 't' || key == 'T') {
            active_tracker_name = (active_tracker_name == "siamrpn") ? "csrt" : "siamrpn";
            std::cout << "\n[ENGINE] Switched tracker engine to: " << active_tracker_name << std::endl;
            if (tracking) {
                selected_bbox = cv::Rect2d(
                    static_cast<double>(smooth_cx - last_bbox_size.width / 2.0),
                    static_cast<double>(smooth_cy - last_bbox_size.height / 2.0),
                    last_bbox_size.width,
                    last_bbox_size.height
                );
                has_selected_bbox = true;
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
