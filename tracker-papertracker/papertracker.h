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
#include "aruco/arucofidmarkers.h"
#include "papertracker-dialog.h"
#include "head.h"
#include "anglecoveragetracker.h"

class PaperTrackerDialog;

class PaperTracker : protected virtual QThread, public ITracker
{
public:
    PaperTracker();
    ~PaperTracker() override;
    module_status start_tracker(QFrame *) override;
    void data(double *data) override;
    void run() override;
    bool tracking_started() const;
    void update_settings();
    bool restart_required() const;

private:
    struct marker_detection_info : public std::vector<cv::Point2f> {
        int id;

        marker_detection_info(int id, const std::vector<cv::Point2f> &corners) : id(id) {
            for (const auto corner : corners)
                push_back(corner);
        }
    };

    papertracker::Head head;
    aruco::MarkerDetector detector;
    papertracker_dictionary current_dictionary;
    std::unique_ptr<video::impl::camera> camera;
    cv::Mat camera_matrix;
    std::vector<double> dist_coeffs;
    cv::Rect2i last_roi;
    bool has_key_marker;
    std::vector<marker_detection_info> detected_markers;
    std::vector<cv::Vec3d> starting_rvecs;
    std::vector<cv::Vec3d> starting_tvecs;
    cv::Vec3d starting_head_origin;
    cv::Vec3d current_head_origin;
    double last_marker_height_cm;
    double last_head_circumference_cm;
    std::unordered_set<int> marker_highlight_set;
    papertracker::AngleCoverageTracker visited_angles;
    papertracker::AngleCoverageBin last_bin;
    papertracker_settings s;
    papertracker_static_settings static_settings;
    std::unique_ptr<cv_video_widget> videoWidget;
    std::unique_ptr<QHBoxLayout> layout;
    Timer fps_timer;
    Timer last_detection_timer;
    double fps = 0;
    double no_detection_timeout = 0;
    QMutex camera_mtx;
    QMutex data_mtx;
    bool started_;
    bool use_fixed_threshold;
    unsigned int adaptive_size_pos;

    bool open_camera();
    bool process_frame(cv::Mat& frame, const cv::Rect2i *roi = nullptr);
    void cycle_threshold_params();
    void set_threshold_params();
    cv::Mat build_camera_matrix(int image_width, int image_height, double diagonal_fov);
    std::vector<int> get_key_markers(const std::vector<marker_detection_info> &detection_info, const std::unordered_map<int, cv::Vec3d> &marker_tvecs);
    cv::Vec3d get_approximate_head_origin(const std::vector<cv::Vec3d> &marker_rvecs, const std::vector<cv::Vec3d> &marker_tvecs);
    cv::Rect2f get_marker_detected_region(const std::vector<marker_detection_info> &markers);
    bool markers_disappeared(const std::vector<int> &expected, const std::vector<marker_detection_info> &detected);
    void draw_head_bounding_box(cv::Mat &image);
    void draw_marker_border(cv::Mat &image, const std::vector<cv::Point2f> &image_points, int id, const cv::Scalar &marker_border = cv::Scalar(0, 0, 255));
    void draw_axes(cv::Mat &image, const cv::Vec3d &rvec, const cv::Vec3d &tvec, double axis_length=1, bool color=true);
    void update_fps();

    friend class PaperTrackerDialog;
};

class PaperTrackerMetadata : public Metadata
{
    Q_OBJECT

    QString name() override { return tr("PaperTracker"); }
    QIcon icon() override { return QIcon(":/images/papertracker.png"); }
};

