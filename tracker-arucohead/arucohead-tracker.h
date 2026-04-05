/* Copyright (c) 2026, Adrian Lopez <adrianlopezroche@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 */
#pragma once

#include <QThread>
#include <QHBoxLayout>
#include <QMutex>
#include <unordered_map>
#include "api/plugin-api.hpp"
#include "cv/video-widget.hpp"
#include "video/camera.hpp"
#include "compat/timer.hpp"
#include "aruco/markerdetector.h"
#include "arucohead-dialog.h"
#include "head.h"
#include "anglecoveragetracker.h"

class arucohead_dialog;

class arucohead_tracker : protected virtual QThread, public ITracker
{
public:
    arucohead_tracker();
    ~arucohead_tracker() override;
    module_status start_tracker(QFrame *) override;
    void data(double *data) override;
    void run() override;

private:
    struct marker_detection_info {
        int id;
        std::vector<cv::Point2f> corners;

        marker_detection_info(int id, const std::vector<cv::Point2f> &corners) : id(id), corners(corners)
        {}
    };

    arucohead::Head head;
    aruco::MarkerDetector detector;
    std::unique_ptr<video::impl::camera> camera;
    cv::Mat camera_matrix;
    std::vector<double> dist_coeffs;
    cv::Rect2f last_roi;
    bool has_marker;
    std::unordered_map<int, cv::Vec3d> previous_marker_rvecs;
    std::vector<marker_detection_info> detected_markers;
    std::unordered_set<int> marker_highlight_set;
    arucohead::AngleCoverageTracker visited_angles;
    arucohead::AngleCoverageBin last_bin;
    settings s;
    std::unique_ptr<cv_video_widget> videoWidget;
    std::unique_ptr<QHBoxLayout> layout;
    Timer fps_timer;
    double fps = 0;
    QMutex camera_mtx;
    QMutex data_mtx;

    bool open_camera();
    bool process_frame(cv::Mat& frame, const cv::Rect2f *roi = nullptr);
    cv::Mat build_camera_matrix(int image_width, int image_height, double diagonal_fov);
    cv::Rect2f get_marker_detected_region(const std::vector<marker_detection_info> &markers);
    bool markers_disappeared(const std::vector<int> &expected, const std::vector<marker_detection_info> &detected);
    void draw_head_bounding_box(cv::Mat &image);
    void draw_marker_border(cv::Mat &image, const std::vector<cv::Point2f> &image_points, int id, const cv::Scalar &marker_border = cv::Scalar(0, 0, 255));
    void draw_axes(cv::Mat &image, const cv::Vec3d &rvec, const cv::Vec3d &tvec, double axis_length=1, bool color=true);
    void update_fps();

    friend class arucohead_dialog;
};

class arucohead_metadata : public Metadata
{
    Q_OBJECT

    QString name() override { return tr("ArUcoHead paper marker tracker"); }
    QIcon icon() override { return QIcon(":/images/arucohead.png"); }
};

