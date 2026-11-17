import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workflow"))
import src.figure_style as fs  # noqa: E402


def test_apply_style_embeds_truetype_fonts_and_hides_top_right_spines():
    with matplotlib.rc_context():
        fs.apply_style()
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert not matplotlib.rcParams["axes.spines.top"]
        assert not matplotlib.rcParams["axes.spines.right"]
        assert matplotlib.rcParams["xtick.labelsize"] >= 7


def test_figure_size_is_180_mm_wide():
    width_in, height_in = fs.figure_size(90)
    assert abs(width_in * 25.4 - 180) < 1e-9
    assert abs(height_in * 25.4 - 90) < 1e-9


def test_save_figure_writes_png_and_pdf_and_keeps_figure_open(tmp_path):
    fig, ax = plt.subplots(figsize=fs.figure_size(40))
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "sub" / "fig.png"
    fs.save_figure(fig, out)
    assert out.stat().st_size > 0
    assert out.with_suffix(".pdf").stat().st_size > 0
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_every_measure_has_a_distinct_colour_and_marker():
    styles = [fs.MEASURE_STYLE[m] for m in fs.MEASURE_ORDER]
    assert len({c for c, _ in styles}) == len(styles)
    assert len({mk for _, mk in styles}) == len(styles)


def test_event_colours_are_distinct_okabe_ito_and_double_encoded():
    styles = list(fs.EVENT_STYLE.values())
    assert {s["color"] for s in styles} <= set(fs.OKABE_ITO)
    for key in ("color", "linestyle", "marker"):
        assert len({s[key] for s in styles}) == len(styles)
    assert fs.EVENT_LABELS["river"] == "River flood"
