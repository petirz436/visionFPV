#include "siamrpn_tracker.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

SiamRPNTracker::SiamRPNTracker() {
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }
    pModule_ = PyImport_ImportModule("cv2");
    if (pModule_) {
        pDict_ = PyModule_GetDict(pModule_);
    } else {
        std::cerr << "[ERROR SiamRPN C++] Failed to import cv2 Python module." << std::endl;
    }
}

SiamRPNTracker::~SiamRPNTracker() {
    if (pTracker_) {
        Py_DECREF(pTracker_);
        pTracker_ = nullptr;
    }
    if (pModule_) {
        Py_DECREF(pModule_);
        pModule_ = nullptr;
    }
}

bool SiamRPNTracker::load_models(const std::string& models_dir) {
    std::string dir = models_dir;
    if (!std::ifstream(dir + "/dasiamrpn_model.onnx").good()) {
        if (std::ifstream("models/dasiamrpn_model.onnx").good()) {
            dir = "models";
        } else if (std::ifstream("../models/dasiamrpn_model.onnx").good()) {
            dir = "../models";
        }
    }

    std::string m1 = dir + "/dasiamrpn_model.onnx";
    std::string m2 = dir + "/dasiamrpn_kernel_cls1.onnx";
    std::string m3 = dir + "/dasiamrpn_kernel_r1.onnx";

    std::ifstream f1(m1), f2(m2), f3(m3);
    if (!f1.good() || !f2.good() || !f3.good()) {
        std::cerr << "[WARNING SiamRPN C++] ONNX models not found in '" << dir << "'." << std::endl;
        models_loaded_ = false;
        return false;
    }

    if (!pDict_) return false;

    std::string build_script = 
        "import cv2\n"
        "import os\n"
        "m1 = os.path.abspath('" + m1 + "')\n"
        "m2 = os.path.abspath('" + m2 + "')\n"
        "m3 = os.path.abspath('" + m3 + "')\n"
        "net_siam = cv2.dnn.readNet(m1)\n"
        "net_cls = cv2.dnn.readNet(m2)\n"
        "net_r1 = cv2.dnn.readNet(m3)\n"
        "tracker = cv2.TrackerDaSiamRPN_create(net_siam, net_cls, net_r1)\n";

    PyObject* pCode = PyRun_String(build_script.c_str(), Py_file_input, pDict_, pDict_);
    if (!pCode) {
        PyErr_Print();
        models_loaded_ = false;
        return false;
    }
    Py_DECREF(pCode);

    PyObject* pTrk = PyDict_GetItemString(pDict_, "tracker");
    if (!pTrk) {
        models_loaded_ = false;
        return false;
    }

    if (pTracker_) Py_DECREF(pTracker_);
    pTracker_ = pTrk;
    Py_INCREF(pTracker_);

    models_loaded_ = true;
    std::cout << "[SiamRPN C++] All 3 ONNX models loaded successfully!" << std::endl;
    return true;
}

bool SiamRPNTracker::init(const cv::Mat& frame, const cv::Rect2d& bbox) {
    if (!models_loaded_) {
        if (!load_models()) return false;
    }
    if (!pTracker_ || frame.empty()) return false;

    std::string init_script = 
        "import numpy as np\n"
        "h, w, c = frame_bytes_h, frame_bytes_w, 3\n"
        "img = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, c))\n"
        "tracker.init(img, (int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)))\n";

    PyDict_SetItemString(pDict_, "frame_bytes_h", PyLong_FromLong(frame.rows));
    PyDict_SetItemString(pDict_, "frame_bytes_w", PyLong_FromLong(frame.cols));
    PyDict_SetItemString(pDict_, "bbox_x", PyFloat_FromDouble(bbox.x));
    PyDict_SetItemString(pDict_, "bbox_y", PyFloat_FromDouble(bbox.y));
    PyDict_SetItemString(pDict_, "bbox_w", PyFloat_FromDouble(bbox.width));
    PyDict_SetItemString(pDict_, "bbox_h", PyFloat_FromDouble(bbox.height));

    PyObject* pBytes = PyBytes_FromStringAndSize((const char*)frame.data, frame.total() * frame.elemSize());
    PyDict_SetItemString(pDict_, "frame_bytes", pBytes);

    PyObject* pRes = PyRun_String(init_script.c_str(), Py_file_input, pDict_, pDict_);
    Py_DECREF(pBytes);

    if (!pRes) {
        PyErr_Print();
        initialized_ = false;
        return false;
    }
    Py_DECREF(pRes);

    initialized_ = true;
    return true;
}

bool SiamRPNTracker::update(const cv::Mat& frame, cv::Rect2d& bbox) {
    if (!initialized_ || !pTracker_ || frame.empty()) return false;

    std::string update_script = 
        "h, w, c = frame_bytes_h, frame_bytes_w, 3\n"
        "img = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, c))\n"
        "ok, box = tracker.update(img)\n"
        "res_ok = ok\n"
        "res_box = [float(b) for b in box] if ok else [0.0, 0.0, 0.0, 0.0]\n";

    PyDict_SetItemString(pDict_, "frame_bytes_h", PyLong_FromLong(frame.rows));
    PyDict_SetItemString(pDict_, "frame_bytes_w", PyLong_FromLong(frame.cols));

    PyObject* pBytes = PyBytes_FromStringAndSize((const char*)frame.data, frame.total() * frame.elemSize());
    PyDict_SetItemString(pDict_, "frame_bytes", pBytes);

    PyObject* pRes = PyRun_String(update_script.c_str(), Py_file_input, pDict_, pDict_);
    Py_DECREF(pBytes);

    if (!pRes) {
        PyErr_Print();
        return false;
    }
    Py_DECREF(pRes);

    PyObject* pOk = PyDict_GetItemString(pDict_, "res_ok");
    bool ok = (pOk == Py_True);
    if (ok) {
        PyObject* pBox = PyDict_GetItemString(pDict_, "res_box");
        if (pBox && PyList_Check(pBox) && PyList_Size(pBox) == 4) {
            bbox.x = PyFloat_AsDouble(PyList_GetItem(pBox, 0));
            bbox.y = PyFloat_AsDouble(PyList_GetItem(pBox, 1));
            bbox.width = PyFloat_AsDouble(PyList_GetItem(pBox, 2));
            bbox.height = PyFloat_AsDouble(PyList_GetItem(pBox, 3));
        }
    }
    return ok;
}
