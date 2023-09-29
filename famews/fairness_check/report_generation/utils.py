import io
import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Image, Paragraph, Table, TableStyle
from reportlab.platypus.flowables import Image as ImageType
from reportlab.platypus.tables import Table as TableType

TABLE_STYLE = [
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
    ("BOX", (0, 0), (-1, -1), 1, colors.black),
]

TOC_STYLE = [
    ParagraphStyle(
        fontName="Courier-Bold",
        fontSize=14,
        name="TOCHeading1",
        leftIndent=30,
        firstLineIndent=-20,
        spaceBefore=5,
        leading=16,
    ),
    ParagraphStyle(
        fontName="Courier",
        fontSize=12,
        name="TOCHeading2",
        leftIndent=50,
        firstLineIndent=-20,
        spaceBefore=0,
        leading=12,
    ),
    ParagraphStyle(
        fontName="Courier",
        fontSize=10,
        name="TOCHeading3",
        leftIndent=70,
        firstLineIndent=-20,
        spaceBefore=0,
        leading=12,
    ),
]


class MyDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kw):
        self.allowSplitting = 0
        BaseDocTemplate.__init__(self, filename, **kw)

    def afterFlowable(self, flowable):
        "Registers TOC entries."
        if flowable.__class__.__name__ == "Paragraph":
            style = flowable.style.name
            if style == "Heading1":
                level = 0
            elif style == "Heading2":
                level = 1
            elif style == "sub_title":
                level = 2
            else:
                return
            text = flowable.getPlainText()
            self.notify("TOCEntry", tuple((level, text, self.page)))


def fig2image(fig: Figure) -> ImageType:
    """Convert a matplotlib image into a reportlab image.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure

    Returns
    -------
    ImageType
        Reportlab image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    x, y = fig.get_size_inches()
    plt.close(fig)
    return Image(buf, x * inch, y * inch)


def double_fig2image(fig1: Figure, fig2: Figure) -> TableType:
    """Produce an image with two figures side by side.

    Parameters
    ----------
    fig1 : Figure
        First Matplotlib figure
    fig2 : Figure
        Second Matplotlib figure

    Returns
    -------
    TableType
        Table contaning both images
    """
    chart_style = TableStyle(
        [("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "CENTER")]
    )
    x, y = fig1.get_size_inches()
    return Table(
        [[fig2image(fig1), fig2image(fig2)]],
        colWidths=[x * inch, x * inch],
        rowHeights=[y * inch],
        style=chart_style,
    )


def df2table(df: pd.DataFrame, table_style: list = TABLE_STYLE) -> TableType:
    """Convert a pandas Dataframe to a reportlab table.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe

    Returns
    -------
    TableType
        Table
    """
    return Table(
        [[Paragraph(col) for col in df.columns]] + df.values.tolist(),
        style=table_style,
        hAlign="LEFT",
    )


def round_down(x: float) -> float:
    """Round x to the closest smaller float at 0.05.

    Parameters
    ----------
    x : float
        value to round

    Returns
    -------
    float
        Rounded value
    """
    return math.floor(x / 0.05) * 0.05


def round_up(x: float) -> float:
    """Round x to the closest greater float at 0.05.

    Parameters
    ----------
    x : float
        value to round

    Returns
    -------
    float
        Rounded value
    """
    return math.ceil(x / 0.05) * 0.05
