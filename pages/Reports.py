from __future__ import annotations

from datetime import date

import plotly.express as px
import streamlit as st

from services.attendance_service import attendance_service
from services.export_service import export_service
from utils.ui import section_title


def render(current_user: dict[str, object]) -> None:
    section_title("Reports")
    start_date = st.date_input("Start Date", value=date.today().replace(day=1))
    end_date = st.date_input("End Date", value=date.today())
    status = st.selectbox("Status", ["", "checked_in", "checked_out", "late"])

    employees = attendance_service.employees_queryset()
    departments = attendance_service.departments_queryset()
    employee_options = ["All Employees"] + [f"{emp.employee_id} - {emp.name}" for emp in employees]
    department_options = ["All Departments"] + [dept.name for dept in departments]
    employee_choice = st.selectbox("Employee", employee_options)
    department_choice = st.selectbox("Department", department_options)

    employee_id = None
    department_id = None
    if employee_choice != "All Employees":
        code = employee_choice.split(" - ", 1)[0]
        employee_id = next((emp.id for emp in employees if emp.employee_id == code), None)
    if department_choice != "All Departments":
        department_id = next((dept.id for dept in departments if dept.name == department_choice), None)

    df = attendance_service.attendance_history(
        employee_id=employee_id,
        department_id=department_id,
        status=status or None,
        start_date=start_date,
        end_date=end_date,
    )

    st.dataframe(df, use_container_width=True, hide_index=True)

    if not df.empty:
        st.subheader("Attendance by Day")
        day_df = df.groupby("attendance_date").size().reset_index(name="count")
        st.plotly_chart(px.bar(day_df, x="attendance_date", y="count", title="Attendance Activity"), use_container_width=True)

        csv_path = export_service.export_csv(df, f"attendance_{start_date}_{end_date}.csv")
        xlsx_path = export_service.export_excel(df, f"attendance_{start_date}_{end_date}.xlsx")
        pdf_path = export_service.export_pdf(df, "Attendance Report", f"attendance_{start_date}_{end_date}.pdf")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Download CSV", data=csv_path.read_bytes(), file_name=csv_path.name, mime="text/csv", use_container_width=True)
        with col2:
            st.download_button(
                "Download Excel",
                data=xlsx_path.read_bytes(),
                file_name=xlsx_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with col3:
            st.download_button("Download PDF", data=pdf_path.read_bytes(), file_name=pdf_path.name, mime="application/pdf", use_container_width=True)
    else:
        st.info("No attendance records found for the selected filters.")
