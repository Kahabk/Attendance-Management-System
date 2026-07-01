from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import settings


class ExportService:
    def _ensure_export_dir(self) -> Path:
        settings.export_dir.mkdir(parents=True, exist_ok=True)
        return settings.export_dir

    def list_exports(self, suffix: str | None = None) -> list[Path]:
        export_dir = self._ensure_export_dir()
        files = sorted((path for path in export_dir.iterdir() if path.is_file()), key=lambda item: item.stat().st_mtime, reverse=True)
        if suffix is None:
            return files
        normalized = suffix.lower().lstrip(".")
        return [path for path in files if path.suffix.lower().lstrip(".") == normalized]

    def delete_export(self, path: Path) -> bool:
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def export_csv(self, df: pd.DataFrame, filename: str) -> Path:
        path = self._ensure_export_dir() / filename
        df.to_csv(path, index=False)
        return path

    def export_excel(self, df: pd.DataFrame, filename: str) -> Path:
        path = self._ensure_export_dir() / filename
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Report")
        return path

    def export_pdf(self, df: pd.DataFrame, title: str, filename: str) -> Path:
        path = self._ensure_export_dir() / filename
        doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
        styles = getSampleStyleSheet()
        story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
        data = [list(df.columns)] + df.astype(str).values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.extend([table, Spacer(1, 12), Paragraph(f"Generated at {datetime.utcnow().isoformat()} UTC", styles["Normal"])])
        doc.build(story)
        return path


export_service = ExportService()
