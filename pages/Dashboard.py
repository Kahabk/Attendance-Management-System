from __future__ import annotations

import plotly.express as px
import streamlit as st

from services.attendance_service import attendance_service
from utils.ui import metric_card, section_title


def render(current_user: dict[str, object]) -> None:
    st.markdown(f"<div class='hero-title'>Welcome, {current_user.get('full_name', 'User')}</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Live attendance overview and operational analytics</div>", unsafe_allow_html=True)

    metrics = attendance_service.get_dashboard_metrics()
    cols = st.columns(6)
    cards = [
        ("Total Employees", metrics["total_employees"], "Active staff"),
        ("Present Today", metrics["present_today"], "Recognized today"),
        ("Absent Today", metrics["absent_today"], "Not detected"),
        ("Late Arrivals", metrics["late_arrivals"], "After start time"),
        ("Monthly Attendance %", f'{metrics["monthly_attendance_percentage"]:.1f}%', "Current month"),
        ("Registered Faces", metrics["registered_faces"], "Stored embeddings"),
    ]
    for col, (label, value, note) in zip(cols, cards, strict=False):
        with col:
            st.markdown(metric_card(label, str(value), note), unsafe_allow_html=True)

    section_title("Attendance Trends")
    daily = attendance_service.get_daily_series(14)
    weekly = attendance_service.get_weekly_series(8)
    monthly = attendance_service.get_monthly_series(12)
    dept = attendance_service.department_statistics()

    chart_cols = st.columns(2)
    with chart_cols[0]:
        if not daily.empty:
            fig = px.line(daily, x="date", y="present", title="Daily Attendance", markers=True)
            st.plotly_chart(fig, width="stretch")
    with chart_cols[1]:
        if not weekly.empty:
            fig = px.bar(weekly, x="week", y="present", title="Weekly Attendance", text_auto=True)
            st.plotly_chart(fig, width="stretch")

    chart_cols2 = st.columns(2)
    with chart_cols2[0]:
        if not monthly.empty:
            fig = px.area(monthly, x="month", y="present", title="Monthly Attendance")
            st.plotly_chart(fig, width="stretch")
    with chart_cols2[1]:
        if not dept.empty:
            fig = px.bar(dept, x="department", y="present_today", color="employees", title="Department Statistics", text_auto=True)
            st.plotly_chart(fig, width="stretch")
