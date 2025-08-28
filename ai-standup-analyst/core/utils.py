from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

if TYPE_CHECKING:
    import plotly.graph_objs as go

def create_sample_data(rows: int = 90) -> pd.DataFrame:
    """Generate a fake dataset for demo purposes."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=rows, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "sales": (rng.normal(100, 10, rows).cumsum() / 5 + 200).round(2),
        "customers": np.clip(rng.normal(50, 10, rows), 5, None).round(0),
        "region": rng.choice(["North", "South", "East", "West"], size=rows),
        "product": rng.choice(["Product A", "Product B", "Product C"], size=rows),
    })
    # Add a couple “interesting” anomalies
    df.loc[rows // 3, "sales"] *= 1.8
    df.loc[(rows // 2) + 5, "customers"] *= 2
    return df

def _try_export_fig(fig, path: str) -> bool:
    try:
        import plotly.io as pio
        pio.write_image(fig, path, width=900, height=500, scale=2)  # needs kaleido
        return True
    except Exception:
        return False

def generate_pdf_report(
    insights: List[Dict[str, Any]],
    jokes: List[str],
    data_shape: tuple,
    figs: Optional[List[Tuple[str, "go.Figure"]]] = None,
) -> Optional[str]:
    """Generate a PDF with insights, jokes, and charts."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_comedy_report_{ts}.pdf"
    try:
        styles = getSampleStyleSheet()
        title = ParagraphStyle("T", parent=styles["Title"], textColor="#8E0F18")
        body = styles["BodyText"]

        story = [
            Paragraph("AI Stand-Up Analyst — Weekly Roast", title),
            Spacer(1, 6),
            Paragraph(f"Dataset: {data_shape[0]} rows × {data_shape[1]} cols", body),
            Spacer(1, 12),
        ]

        for i, (ins, joke) in enumerate(zip(insights, jokes), start=1):
            story += [
                Paragraph(f"<b>Insight #{i}</b>", styles["Heading2"]),
                Paragraph(ins["summary"], body),
                Paragraph(f"🎤 {joke}", body),
                Spacer(1, 8),
            ]

        # Add charts if provided
        if figs:
            story.append(Paragraph("<b>Charts</b>", styles["Heading2"]))
            for title_txt, fig in figs:
                tmp_png = f"__chart_{ts}_{abs(hash(title_txt)) % 10000}.png"
                if _try_export_fig(fig, tmp_png):
                    story += [
                        Paragraph(title_txt, body),
                        Image(tmp_png, width=480, height=260),
                        Spacer(1, 8),
                    ]

        doc = SimpleDocTemplate(filename, pagesize=letter)
        doc.build(story)
        return filename
    except Exception:
        return None
