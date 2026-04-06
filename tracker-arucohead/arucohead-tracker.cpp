/* Copyright (c) 2026, Adrian Lopez <adrianlopezroche@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 */
#include "arucohead-tracker.h"
#include "arucohead-util.h"
#include "api/plugin-api.hpp"
#include "cv/init.hpp"
#include "compat/sleep.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "aruco_nano.h"
#include "config.h"
#include <unordered_map>
#include <unordered_set>
#include <QDebug>

using namespace arucohead;

/* Open camera selected in settings.
*/
bool arucohead_tracker::open_camera()
{
    /* Prevent concurrent access to the camera.
    */
    QMutexLocker l(&camera_mtx);

    /* Create camera object.
    */
    QString camera_name = s.camera_name;

    if (s.camera_name == "")
        camera_name = video::camera_names()[0];

    camera = video::make_camera(camera_name);
    if (!camera)
        return false;

    /* Start camera.
    */
    video::impl::camera::info args {};
    args.width = s.frame_width;
    args.height = s.frame_height;
    args.fps = s.fps;
    args.use_mjpeg = s.use_mjpeg;

    if (!camera->start(args)) {
        qDebug() << "ArUcoHead: could not open camera.";
        return false;
    }

    return true;
}

/* Detect markers, update head pose, and draw AR elements.
*/
bool arucohead_tracker::process_frame(cv::Mat &frame, const cv::Rect2f *roi)
{
    /* Pose vectors for each detected marker.
    */
    std::unordered_map<int, cv::Vec3d> marker_rvecs;
    std::unordered_map<int, cv::Vec3d> marker_tvecs;

    /* Head pose vectors for each detected marker.
    */
    std::unordered_map<int, cv::Vec3d> pose_rvecs;
    std::unordered_map<int, cv::Vec3d> pose_tvecs;

    /* Marker reliability metrics.
    */
    std::unordered_map<int, double> marker_error_ratios;
    std::unordered_map<int, double> marker_z_angles;
    std::unordered_set<int> excluded_markers;

    /* Marker corners (in object space).
    */
    const double marker_size_cm = s.aruco_marker_size_mm / 10.0;

    const cv::Mat objectPoints = (cv::Mat_<double>(4, 3) <<
        -marker_size_cm / 2.0,  marker_size_cm / 2.0, 0.0,
         marker_size_cm / 2.0,  marker_size_cm / 2.0, 0.0,
         marker_size_cm / 2.0, -marker_size_cm / 2.0, 0.0,
        -marker_size_cm / 2.0, -marker_size_cm / 2.0, 0.0
    );

    /* Create ROI image.
    */
    cv::Mat image;

    if (roi) {
        if (roi->width > 0 && roi->height > 0) {
            image = frame(*roi & cv::Rect2f(0, 0, frame.cols, frame.rows));
        } else {
            image = frame;
            roi = nullptr;
        }
    } else {
        image = frame;
    }

    /* Detect markers and draw their borders.
    */
    detected_markers.clear();

    if (s.aruco_dictionary == arucohead_dictionary::ARUCOHEAD_DICT_ARUCO_ORIGINAL) {
        /* Use OpenTrack's ArUco fork for original ArUco dictionary.
        */
        std::vector<aruco::Marker> markers;
        detector.detect(image, markers, cv::Mat(), cv::Mat(), -1, false);

        const int min_marker_id = s.first_marker_id;
        const int max_marker_id = s.first_marker_id + s.number_of_markers - 1;

        for (const auto &marker : markers) {
            if (marker.id < min_marker_id || marker.id > max_marker_id)
                continue;

            detected_markers.push_back(marker_detection_info(marker.id, marker));
        }
    } else {
        /* Use ArUco Nano for MIP 36h12 and AprilTag 36h11 dictionaries.
        */
        auto markers = aruconano::MarkerDetector::detect(image, 10U, s.aruco_dictionary == arucohead_dictionary::ARUCOHEAD_DICT_ARUCO_MIP_36h12 ? aruconano::MarkerDetector::ARUCO_MIP_36h12 : aruconano::MarkerDetector::APRILTAG_36h11);

        const int min_marker_id = s.first_marker_id;
        const int max_marker_id = s.first_marker_id + s.number_of_markers - 1;

        for (const auto &marker : markers) {
            if (marker.id < min_marker_id || marker.id > max_marker_id)
                continue;

            detected_markers.push_back(marker_detection_info(marker.id, marker));
        }
    }

    /* Express corners in terms of original frame, if necessary.
    */
    if (roi != nullptr) {
        for (auto &marker : detected_markers) {
            for (auto &corner : marker.corners) {
                corner.x += roi->x;
                corner.y += roi->y;
            }
        }
    }

    /* If no markers detected and ROI is not null, signal need to reset ROI.
    */
    bool reset_roi;

    if (detected_markers.size() == 0 && roi != nullptr)
        return false;
    else
        reset_roi = false;

    last_roi = get_marker_detected_region(detected_markers);

    /* Caclulate poses for detected markers.
    */
    for (size_t i = 0; i < detected_markers.size(); ++i) {
        std::vector<cv::Vec3d> rvecs;
        std::vector<cv::Vec3d> tvecs;
        std::vector<double> reprojection_errors;

        if (cv::solvePnPGeneric(objectPoints, detected_markers[i].corners, camera_matrix, dist_coeffs, rvecs, tvecs, false, cv::SOLVEPNP_IPPE, cv::noArray(), cv::noArray(), reprojection_errors)) {
            const int id = detected_markers[i].id;

            /* Choose the best solution among the two candidates. */
            const double error0 = reprojection_errors[0];
            const double error1 = reprojection_errors[1];
            const double error_ratio = std::min(error0, error1) / std::max(error0, error1);

            const int solution_index = error0 <= error1 ? 0 : 1;

            /* Detect and correct for sudden flips in a marker orientation when pose solutions are ambiguous.
            */
            if (error_ratio < ARUCOHEAD_MARKER_REPROJECTION_ERROR_THRESHOLD) {
                // We have a good pose, so use it (or else we might get stuck on a wrong solution).
                marker_rvecs[id] = rvecs[solution_index];
                marker_tvecs[id] = tvecs[solution_index];
            } else if (previous_marker_rvecs.count(id) != 0 && marker_has_flipped(previous_marker_rvecs[id], rvecs[solution_index])) {
                // Marker flip detected, so correct it by choosing the pose closest to the previous one.
                if (angle_between_rotations(previous_marker_rvecs[id], rvecs[0]) < angle_between_rotations(previous_marker_rvecs[id], rvecs[1])) {
                    marker_rvecs[id] = rvecs[0];
                    marker_tvecs[id] = tvecs[0];
                } else {
                    marker_rvecs[id] = rvecs[1];
                    marker_tvecs[id] = tvecs[1];
                }
            } else {
                // No flip, so use pose with the least reprojection error even if not ideal.
                marker_rvecs[id] = rvecs[solution_index];
                marker_tvecs[id] = tvecs[solution_index];
            }

            /* Collect reliability metrics.
            */
            marker_error_ratios[id] = error_ratio;
            marker_z_angles[id] = get_marker_z_angle(marker_rvecs[id]);

            /* Exclude markers that are too ambiguous, too close to head-on, or too oblique.
            */
            if (error_ratio > ARUCOHEAD_MARKER_EXCLUSION_AMBIGUITY_THRESHOLD ||
                marker_z_angles[id] < CV_PI / 180.0 * s.marker_min_angle ||
                marker_z_angles[id] > CV_PI / 180.0 * s.marker_max_angle)
            {
                excluded_markers.insert(id);
            }

            /* Save current marker rvec for next frame.
            */
            previous_marker_rvecs[id] = marker_rvecs[id];
        }
    }

    /* Prune away unreliable markers unless none would remain.
     */
    std::vector<marker_detection_info> selected_markers;

    marker_highlight_set.clear();

    const size_t excluded_marker_count = excluded_markers.size();

    if (excluded_marker_count > 0) {
        if (excluded_marker_count < detected_markers.size()) {
            /* Keep only reliable markers.
            */
            for (size_t i = 0; i < detected_markers.size(); ++i) {
                if (excluded_markers.count(detected_markers[i].id) == 0) {
                    selected_markers.push_back(detected_markers[i]);
                    marker_highlight_set.insert(detected_markers[i].id);
                }
            }
        } else if (detected_markers.size() > 0) {
            /* Fallback: no good markers, so choose among the remaining markers.
            */
            std::sort(detected_markers.begin(), detected_markers.end(), [&marker_z_angles, this](const auto &a, const auto &b) {
                const bool head_has_a = head.has_handle(a.id);
                const bool head_has_b = head.has_handle(b.id);

                if (head_has_a && !head_has_b)
                    return true;
                else if (!head_has_a && head_has_b)
                    return false;

                const double sorting_angle_a = fabs(marker_z_angles[a.id] - CV_PI / 180.0 * s.marker_min_angle);
                const double sorting_angle_b = fabs(marker_z_angles[b.id] - CV_PI / 180.0 * s.marker_min_angle);

                return sorting_angle_a < sorting_angle_b;
            });

            /* Use the first marker.
            */
            selected_markers = { detected_markers[0] };

            marker_highlight_set.insert(detected_markers[0].id);
        }
    } else {
        selected_markers = detected_markers;

        for (const auto &marker : detected_markers)
            marker_highlight_set.insert(marker.id);
    }

    /* Find first marker if none has yet been detected.
    */
    if (!has_marker) {
        if (!selected_markers.empty()) {
            double best_marker_x = circumference_to_radius(s.head_circumference_cm) * 2;
            double best_marker_z = 0;
            cv::Vec3d best_marker_rvec;
            cv::Vec3d best_marker_tvec;
            int best_marker_id = 0;

            for (size_t i = 0; i < selected_markers.size(); ++i) {
                const int id = selected_markers[i].id;

                if (marker_rvecs.count(id) > 0) {
                    auto v = get_xz_direction_vector(marker_rvecs[id]);
                    auto c = circle_edge_intersection(circumference_to_radius(s.head_circumference_cm), v);

                    if (cv::abs(c[0]) < cv::abs(best_marker_x)) {
                        has_marker = true;
                        best_marker_x = c[0];
                        best_marker_z = c[1];
                        best_marker_rvec = marker_rvecs[id];
                        best_marker_tvec = marker_tvecs[id];
                        best_marker_id = id;
                    }
                }
            }

            if (has_marker) {
                cv::Vec3d reference_rvec(CV_PI, 0, 0);

                cv::Vec3d reference_tvec = best_marker_tvec;
                reference_tvec[0] += best_marker_x;
                reference_tvec[1] += s.marker_height_cm;
                reference_tvec[2] += best_marker_z;

                auto [rvec_local, tvec_local] = get_marker_local_transform(best_marker_rvec, best_marker_tvec, reference_rvec, reference_tvec, circumference_to_radius(s.head_circumference_cm), s.marker_height_cm);

                head.set_handle(Marker(best_marker_id, MeanVector(rvec_local, MeanVector::VectorType::ROTATION), MeanVector(tvec_local, MeanVector::VectorType::POLAR)));
            }
        }
    }
    
    if (has_marker && !selected_markers.empty()) {
        /* Compute poses for known markers.
        */
        for (size_t i = 0; i < selected_markers.size(); ++i) {
            const int id = selected_markers[i].id;

            if (head.has_handle(id)) {
                const auto sample_count = head.get_handle(id).rvec_local.get_max_sample_count();

                if (sample_count < ARUCOHEAD_MIN_VECTOR_SAMPLES && selected_markers.size() != 1)
                    continue;

                if (marker_rvecs.count(id) > 0)
                    std::tie(pose_rvecs[id], pose_tvecs[id]) = head.get_pose_from_handle_transform(id, marker_rvecs[id], marker_tvecs[id], circumference_to_radius(s.head_circumference_cm), s.marker_height_cm);
            }
        }

        /* Add/update handles.
        */
        if (pose_rvecs.size() > 0) {
            /* Update head pose.
            */
            {
                QMutexLocker l(&data_mtx);
                head.rvec = average_rotation(pose_rvecs);
                head.tvec = average_translation(pose_tvecs);
            }

            /* Compute local transforms for each marker, adding or updating handles as needed.
            */
            for (size_t i = 0; i < selected_markers.size(); ++i) {
                const int id = selected_markers[i].id;

                if (marker_rvecs.count(id) > 0) {
                    if (!head.has_handle(id)) {
                        auto [rvec_local, tvec_local] = head.get_marker_local_transform(marker_rvecs[id], marker_tvecs[id], circumference_to_radius(s.head_circumference_cm), s.marker_height_cm);
                        head.set_handle(Marker(id, MeanVector(rvec_local, MeanVector::VectorType::ROTATION), MeanVector(tvec_local, MeanVector::VectorType::POLAR)));
                    } else if (pose_rvecs.size() > 1) {
                        auto &handle = head.get_handle(id);

                        if (!handle.rvec_local.outliers_removed() && handle.rvec_local.sample_count() == handle.rvec_local.get_max_sample_count())
                            handle.rvec_local.remove_outliers();

                        if (!handle.tvec_local.outliers_removed() && handle.tvec_local.sample_count() == handle.tvec_local.get_max_sample_count())
                            handle.tvec_local.remove_outliers();

                        if (handle.rvec_local.sample_count() < handle.rvec_local.get_max_sample_count()) {
                            const auto pose_rvec = average_rotation(pose_rvecs, id);
                            const auto pose_tvec = average_translation(pose_tvecs, id);

                            auto [rvec_local, tvec_local] = get_marker_local_transform(marker_rvecs[id], marker_tvecs[id], pose_rvec, pose_tvec, circumference_to_radius(s.head_circumference_cm), s.marker_height_cm);

                            handle.rvec_local.update(rvec_local);
                            handle.tvec_local.update(tvec_local);
                        }
                    }
                }
            }
        }

        /* Check detected markers against expectations and signal need to reset ROI if these don't match.
        */
        cv::Mat R;
        cv::Rodrigues(head.rvec, R);
        const auto euler = rotation_matrix_to_euler_zyx(R);

        const auto bin = visited_angles.get_bin(euler[0], euler[1]);

        if (bin != last_bin && visited_angles.get_visit_count(last_bin) < ARUCOHEAD_ANGLE_COVERAGE_VISIT_THRESHOLD)
            visited_angles.clear_visits(last_bin);

        if (roi == nullptr && visited_angles.get_visit_count(bin) < ARUCOHEAD_ANGLE_COVERAGE_VISIT_THRESHOLD)
            visited_angles.add_visit(bin);

        if (roi != nullptr) {
            if (head.num_handles() < s.number_of_markers && visited_angles.get_visit_count(bin) < ARUCOHEAD_ANGLE_COVERAGE_VISIT_THRESHOLD) {
                reset_roi = true;
            } else {
                auto expected_ids = head.get_expected_visible_ids(CV_PI / 180.0 * s.marker_max_angle);

                if (markers_disappeared(expected_ids, detected_markers))
                    reset_roi = true;
            }
        }
    }

    return !reset_roi;
}

/* Generate a camera matrix for the given image dimension and field of view (in radians).
*/
cv::Mat arucohead_tracker::build_camera_matrix(int image_width, int image_height, double diagonal_fov)
{
    double diagonal_px = cv::sqrt(image_width*image_width + image_height*image_height);

    double focal_length = (diagonal_px / 2) / tan(diagonal_fov / 2);

    double cx = image_width / 2;
    double cy = image_height / 2;

    double data[9] = {
        focal_length, 0,            cx,
        0,            focal_length, cy,
        0,            0,            1
    };

    return cv::Mat(3, 3, CV_64F, data).clone();
}

cv::Rect2f arucohead_tracker::get_marker_detected_region(const std::vector<marker_detection_info> &markers)
{
    if (markers.size() == 0 || markers[0].corners.size() == 0)
        return cv::Rect2f(0, 0, 0, 0);

    cv::Point2f min = markers[0].corners[0];
    cv::Point2f max = markers[0].corners[0];

    for (const auto &marker : markers) {
        for (const auto &corner : marker.corners) {
            min.x = std::min(corner.x, min.x);
            min.y = std::min(corner.y, min.y);
            max.x = std::max(corner.x, max.x);
            max.y = std::max(corner.y, max.y);
        }
    }

    const double w = max.x - min.x;
    const double h = max.y - min.y;

    min.x -= w * ARUCOHEAD_ROI_GROWTH_FACTOR / 2.0;
    min.y -= h * ARUCOHEAD_ROI_GROWTH_FACTOR / 2.0;
    max.x += w * ARUCOHEAD_ROI_GROWTH_FACTOR / 2.0;
    max.y += h * ARUCOHEAD_ROI_GROWTH_FACTOR / 2.0;

    return cv::Rect2f(min, max);
}

bool arucohead_tracker::markers_disappeared(const std::vector<int> &expected, const std::vector<marker_detection_info> &detected) {
    for (const int expected_id : expected) {
        bool found = false;

        for (const auto marker : detected) {
            if (expected_id == marker.id) {
                found = true;
                break;
            }
        }

        if (!found)
            return true;
    }

    return false;
}

/* Draw a bounding box around the head from dimensions defined in settings.
*/
void arucohead_tracker::draw_head_bounding_box(cv::Mat &image)
{
    double head_radius = circumference_to_radius(s.head_circumference_cm);

    std::vector<cv::Point3d> box_points = {
        cv::Point3d(-head_radius, 0, -head_radius),
        cv::Point3d(-head_radius, 0, head_radius),
        cv::Point3d(head_radius, 0, head_radius),
        cv::Point3d(head_radius, 0, -head_radius),
        cv::Point3d(-head_radius, s.marker_height_cm, -head_radius),
        cv::Point3d(-head_radius, s.marker_height_cm, head_radius),
        cv::Point3d(head_radius, s.marker_height_cm, head_radius),
        cv::Point3d(head_radius, s.marker_height_cm, -head_radius)
    };

    std::vector<cv::Point2d> image_points;
    cv::projectPoints(box_points, head.rvec, head.tvec, camera_matrix, dist_coeffs, image_points);

    const cv::Scalar box_color(0, 255, 0);
    const cv::Scalar brightness3(1, 1, 1);
    const cv::Scalar brightness2(0.8, 0.8, 0.8);
    const cv::Scalar brightness1(0.5, 0.5, 0.5);

    const double line_thickness = std::max(image.cols * ARUCOHEAD_LINE_THICKNESS_FRAME_SCALING_FACTOR, 1.0);

    cv::line(image, image_points[0], image_points[1], box_color * brightness2, line_thickness);
    cv::line(image, image_points[1], image_points[2], box_color * brightness3, line_thickness);
    cv::line(image, image_points[2], image_points[3], box_color * brightness2, line_thickness);
    cv::line(image, image_points[3], image_points[0], box_color * brightness1, line_thickness);

    cv::line(image, image_points[4], image_points[5], box_color * brightness2, line_thickness);
    cv::line(image, image_points[5], image_points[6], box_color * brightness3, line_thickness);
    cv::line(image, image_points[6], image_points[7], box_color * brightness2, line_thickness);
    cv::line(image, image_points[7], image_points[4], box_color * brightness1, line_thickness);

    cv::line(image, image_points[0], image_points[4], box_color * brightness1, line_thickness);
    cv::line(image, image_points[1], image_points[5], box_color * brightness3, line_thickness);
    cv::line(image, image_points[2], image_points[6], box_color * brightness3, line_thickness);
    cv::line(image, image_points[3], image_points[7], box_color * brightness1, line_thickness);
}

/* Draw a border around a marker given its vertices in image space.
*/
void arucohead_tracker::draw_marker_border(cv::Mat &image, const std::vector<cv::Point2f> &image_points, int id, const cv::Scalar &border_color)
{
    const size_t vertex_count = image_points.size();

    cv::Point2f center(0, 0);

    const double line_thickness = std::max(image.cols * ARUCOHEAD_LINE_THICKNESS_FRAME_SCALING_FACTOR, 1.0);
    const double font_scale = line_thickness * 0.5;

    for (size_t v = 0; v < vertex_count; ++v) {
        cv::line(image, image_points[v], image_points[(v+1) % vertex_count], border_color, line_thickness);
        center += image_points[v];
    }

    center /= (double)image_points.size();

    std::stringstream ss;
    ss << id;

    const auto text_size = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_SIMPLEX, font_scale, line_thickness, nullptr);

    cv::putText(image, ss.str(), center - cv::Point2f(text_size.width / 2.0, 0), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255) - border_color, line_thickness, cv::LINE_AA);
}

/* Draw coordinate axes for a given pose (rvec/tvec).
*/
void arucohead_tracker::draw_axes(cv::Mat &image, const cv::Vec3d &rvec, const cv::Vec3d &tvec, double axis_length, bool color)
{
    std::vector<cv::Point3d> axis_points = {
        {0, 0, 0},
        {axis_length, 0, 0},
        {0, axis_length, 0},
        {0, 0, axis_length}
    };
    
    std::vector<cv::Point2d> image_points;
    cv::projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);

    const double line_thickness = std::max(image.cols * ARUCOHEAD_LINE_THICKNESS_FRAME_SCALING_FACTOR, 1.0);
        
    cv::line(image, image_points[0], image_points[1], color ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255), line_thickness);
    cv::line(image, image_points[0], image_points[2], color ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 255, 255), line_thickness);
    cv::line(image, image_points[0], image_points[3], color ? cv::Scalar(255, 0, 0) : cv::Scalar(255, 255, 255), line_thickness);
}

/* Update FPS average.
*/
void arucohead_tracker::update_fps()
{
    const double dt = fps_timer.elapsed_seconds();
    fps_timer.start();
    const double alpha = dt/(dt + 0.25);

    if (dt > 1e-3)
    {
        fps *= 1 - alpha;
        fps += alpha * (1./dt + .8);
    }
}

/* Start tracking.
*/
module_status arucohead_tracker::start_tracker(QFrame *videoframe)
{
    videoframe->show();

    videoWidget = std::make_unique<cv_video_widget>(videoframe);
    layout = std::make_unique<QHBoxLayout>();
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(&*videoWidget);
    videoframe->setLayout(&*layout);
    videoWidget->show();

    head.rvec[0] = -CV_PI;
    head.rvec[1] = 0;
    head.rvec[2] = 0;

    start();

    return status_ok();
}

/* Main thread / update loop.
*/
void arucohead_tracker::run() {
    dist_coeffs = std::vector<double>(5, 0);

    if (!open_camera())
        return;

    update_fps();
    
    /* Update loop.
    */
    while (!isInterruptionRequested()) {
        video::frame frame;
        bool new_frame;

        {
            QMutexLocker l(&camera_mtx);

            std::tie(frame, new_frame) = camera->get_frame();

            if (!new_frame) {
                l.unlock();
                portable::sleep(100);
                continue;
            }
        }

        /* Frames must have 1 (grayscale) or 3 (color) channels.
        */
        if (frame.channels != 1 && frame.channels != 3) {
            qDebug() << "ArUcoHead: can't handle" << frame.channels << "color channels";
            return;
        }

        auto frame_mat_temp = cv::Mat(frame.height, frame.width, CV_8UC(frame.channels), (void*)frame.data, frame.stride);

        /* Apply zoom (as region of interest).
        */
        const double zoom = s.zoom / 100.0;
        const int zoomed_width = std::max(frame_mat_temp.size().width / zoom, 1.0);
        const int zoomed_height = std::max(frame_mat_temp.size().height / zoom, 1.0);

        cv::Mat frame_mat = frame_mat_temp(cv::Rect((frame_mat_temp.size().width - zoomed_width)/2, (frame_mat_temp.size().height - zoomed_height)/2, zoomed_width, zoomed_height));

        /* Adjust FOV according to zoom level and set camera matrix.
        */
        const double fov_rad = s.fov * (CV_PI / 180.0);
        const double cropped_fov = 2.0 * atan(tan(fov_rad/2.0) / zoom);
        camera_matrix = build_camera_matrix(frame_mat.size().width, frame_mat.size().height, cropped_fov);

        /* Process the current frame.
        */
        if (!process_frame(frame_mat, &last_roi))
            process_frame(frame_mat);

        /* Draw detected markers.
        */
        for (auto &marker : detected_markers) {
            if (marker_highlight_set.count(marker.id) > 0)
                draw_marker_border(frame_mat, marker.corners, marker.id, cv::Scalar(0, 0, 255));
            else
                draw_marker_border(frame_mat, marker.corners, marker.id, cv::Scalar(0, 0, 170));
        }

        /* Draw head bounding box and coordinate axes.
        */
        if (head.num_handles() > 0) {
            draw_head_bounding_box(frame_mat);
            draw_axes(frame_mat, head.rvec, head.tvec, circumference_to_radius(s.head_circumference_cm) * 1.5);
        }

        /* Update and render FPS indicator.
        */
        update_fps();

        std::stringstream ss;
        ss << "Hz: " << int(fps);

        const double line_thickness = std::max(frame_mat.cols * ARUCOHEAD_LINE_THICKNESS_FRAME_SCALING_FACTOR, 1.0);
        const double font_scale = line_thickness * 0.5;

        const auto text_size = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_SIMPLEX, font_scale, line_thickness, nullptr);

        cv::putText(frame_mat, ss.str(), cv::Point(line_thickness * 5, text_size.height + line_thickness * 5), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 0), line_thickness, cv::LINE_AA);

        /* Show preview.
        */
        if (frame_mat.rows > 0)
            videoWidget->update_image(frame_mat);
    }
}

/* Supply position and orientation data.
*/
void arucohead_tracker::data(double *data)
{
    QMutexLocker l(&data_mtx);

    cv::Mat Rx;
    cv::Mat rvecX = (cv::Mat_<double>(3,1) << CV_PI, 0, 0);
    cv::Rodrigues(rvecX, Rx);

    cv::Mat R;
    cv::Rodrigues(head.rvec, R);
    auto euler = rotation_matrix_to_euler_zyx(R * Rx);

    data[TX] = head.tvec[0];
    data[TY] = -head.tvec[1];
    data[TZ] = head.tvec[2];

    data[Roll] = euler[2] * (180.0 / CV_PI);
    data[Pitch] = -euler[0] * (180.0 / CV_PI);
    data[Yaw] = euler[1] * (180.0 / CV_PI);
}

/* Tracker constructor.
*/
arucohead_tracker::arucohead_tracker() :
    has_marker(false),
    visited_angles(2.0 * CV_PI / ARUCOHEAD_ANGLE_COVERAGE_PITCH_STEPS, 2.0 * CV_PI / ARUCOHEAD_ANGLE_COVERAGE_YAW_STEPS),
    last_roi(0, 0, std::numeric_limits<float>::max(), std::numeric_limits<float>::max()),
    last_bin(std::numeric_limits<int>::max(), std::numeric_limits<int>::max())
{
    opencv_init();
}

/* Tracker destructor.
*/
arucohead_tracker::~arucohead_tracker() {
    requestInterruption();
    wait();
    portable::sleep(1000);
}

/* Settings constructor.
*/
settings::settings() : opts("arucohead-tracker") {};

/* Tracker declaration.
*/
OPENTRACK_DECLARE_TRACKER(arucohead_tracker, arucohead_dialog, arucohead_metadata)
