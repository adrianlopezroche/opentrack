/* Copyright (c) 2026, Adrian Lopez <adrianlopezroche@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 */
#pragma once

#include "ui_arucohead.h"
#include "arucohead-settings.h"
#include "api/plugin-api.hpp"

class arucohead_tracker;

class arucohead_dialog : public ITrackerDialog
{
    Q_OBJECT

public:
    arucohead_dialog();
    void register_tracker(ITracker *) override;
    void unregister_tracker() override;
    bool embeddable() noexcept override { return true; }
    void set_buttons_visible(bool x) override;
    void save() override;
    void reload() override;
private:
    arucohead_settings s;
    Ui::arucohead_dialog ui;
    arucohead_tracker *tracker;
private slots:
    void doOK();
    void doCancel();
    void doOpenCameraSettings();
    void doShowHelp();
};