"""Schematic of how rule 11's calibration, the spin-up and an event run set their length.

Fixed lengths come from config/config.yml; the times that differ per basin
(travel, settle, river event shape, spin-up chunks needed) are illustrative
(ILLUSTRATIVE_HR). The event run's timeline is src.sfincs_run.event_timeline,
as in rule 14, for a compound event with compound.lag_hr auto.

Writes run_lengths_schematic.png/.pdf/.txt to <results_dir>/figures/analysis/.

Usage: python analysis/run_lengths.py
"""

import matplotlib.pyplot as plt
import numpy as np
import yaml
from _common import REPO_ROOT, analysis_path, script_parser
from src.figure_style import (
    EVENT_STYLE,
    apply_style,
    figure_size,
    panel_letters,
    save_figure,
    write_caption,
)
from src.river_forcing import cosine_ramp
from src.sfincs_run import event_timeline

ILLUSTRATIVE_HR = {
    "t_walled": 14.0,  # rule 11: walled travel time, crossing to network end
    "t_settle": 11.5,  # rule 11: settle time L^2/D
    "spinup_chunks": 2,  # rule 13a: chunks until the steady check passes
    "river_rise": 50.0,  # rule 14: Q_start crossing to peak at the inflow
    "river_ramp": 13.0,  # rule 14: ramp from the mean to Q_start
    "river_fall": 40.0,  # rule 14: peak to Q_start again
    "river_travel": 8.0,  # rule event_timing: inflow to the main mouth
    "surge_travel": 2.0,  # rule event_timing: boundary to the main mouth
}

SEA = EVENT_STYLE["coast"]
RIVER = EVENT_STYLE["river"]
SEA_Y0, RIVER_Y0 = 0.0, 2.2  # baselines of the two normalised traces


def _config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "config" / "config.yml").read_text(encoding="utf-8")
    )


def _bracket(ax, x0: float, x1: float, y: float, text: str, ha: str = "center") -> None:
    ax.annotate(
        "",
        xy=(x0, y),
        xytext=(x1, y),
        arrowprops=dict(arrowstyle="<->", lw=0.6, color="0.25", shrinkA=0, shrinkB=0),
    )
    x = {"center": 0.5 * (x0 + x1), "left": x0, "right": x1}[ha]
    ax.text(x, y - 0.08, text, ha=ha, va="top", fontsize=7)


def _traces(ax, t, sea, river, x_label: str) -> None:
    ax.plot(t, SEA_Y0 + sea, color=SEA["color"], linestyle=SEA["linestyle"])
    ax.plot(t, RIVER_Y0 + river, color=RIVER["color"], linestyle=RIVER["linestyle"])
    for y, label, style in (
        (SEA_Y0, "Sea level", SEA),
        (RIVER_Y0, "River\ndischarge", RIVER),
    ):
        ax.text(
            -0.01,
            y + 0.4,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
            color=style["color"],
        )
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel(x_label)


def calibration_panel(ax, cfg: dict) -> None:
    rdm = cfg["river_processing"]["river_depth_modelling"]
    ramp, margin = float(rdm["ramp_hr"]), float(rdm["calibration_margin_hr"])
    factor = float(rdm["run_length"]["settle_factor"])
    t_walled, t_settle = ILLUSTRATIVE_HR["t_walled"], ILLUSTRATIVE_HR["t_settle"]
    t_run = max(t_walled, factor * t_settle)
    run_hr = float(np.ceil(ramp + t_run + margin))
    t = np.linspace(0.0, run_hr, 400)
    w = cosine_ramp(t, 0.0, ramp)
    _traces(
        ax,
        t,
        0.2 + 0.8 * w,
        0.2 + 0.8 * w,
        "Hours since the start of the calibration run",
    )
    for y0, end in (
        (SEA_Y0, "storm tide at the coastal protection RP"),
        (RIVER_Y0, "discharge at the riverine protection RP"),
    ):
        ax.text(
            run_hr, y0 + 1.05, f"held at the {end}", ha="right", va="bottom", fontsize=7
        )
    y = -0.35
    _bracket(ax, 0.0, ramp, y, f"ramp_hr\n{ramp:g} h", ha="left")
    _bracket(
        ax,
        ramp,
        ramp + t_run,
        y,
        f"max(T_walled, {factor:g} x t_settle)\nwave arrival or settling, crossing to sea",
    )
    _bracket(ax, ramp + t_run, run_hr, y, f"calibration_margin_hr\n{margin:g} h")
    ax.text(
        run_hr,
        -1.25,
        f"run = ceil(sum), at most calibration_max_days ({float(rdm['calibration_max_days']):g} d)",
        ha="right",
        va="top",
        fontsize=7,
    )
    ax.set_ylim(-1.55, RIVER_Y0 + 1.4)


def spinup_panel(ax, cfg: dict) -> None:
    sp = cfg["sfincs"]["spinup"]
    chunk, ramp = float(sp["chunk_hr"]), float(sp["boundary_ramp_hr"])
    window = float(
        cfg["river_processing"]["river_depth_modelling"]["steady_check"]["window_hr"]
    )
    k = int(ILLUSTRATIVE_HR["spinup_chunks"])
    end = k * chunk
    t = np.linspace(0.0, end, 400)
    sea = 0.45 - 0.15 * cosine_ramp(t, 0.0, ramp)
    _traces(ax, t, sea, np.full_like(t, 0.3), "Hours since the start of the spin-up")
    ax.text(
        end,
        SEA_Y0 + 0.5,
        "calm sea + SLR x fingerprint",
        ha="right",
        va="bottom",
        fontsize=7,
    )
    ax.text(end, RIVER_Y0 + 0.4, "mean discharge", ha="right", va="bottom", fontsize=7)
    for i in range(1, k + 1):
        x = i * chunk
        ax.axvline(x, color="0.6", lw=0.6, linestyle="--")
        ax.axvspan(x - window, x, color="0.9", lw=0)
        ok = i == k
        ax.text(
            x - window / 2,
            RIVER_Y0 + 1.15,
            "steady" if ok else "not\nsteady",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    y = -0.35
    for i in range(k):
        _bracket(ax, i * chunk, (i + 1) * chunk, y, f"chunk_hr\n{chunk:g} h")
    _bracket(ax, 0.0, ramp, SEA_Y0 + 1.05, f"boundary_ramp_hr {ramp:g} h", ha="left")
    ax.text(
        end,
        -1.25,
        f"spinup_hr = chunks x chunk_hr, until the steady check passes "
        f"(grey: last {window:g} h), at most max_days ({float(sp['max_days']):g} d)",
        ha="right",
        va="top",
        fontsize=7,
    )
    ax.set_ylim(-1.55, RIVER_Y0 + 1.4)


def event_panel(ax, cfg: dict) -> None:
    hg = cfg["boundary_forcings"]["surge"]["hydrograph"]
    surge_ramp, window = float(hg["ramp_hr"]), float(hg["window_hr"])
    fall_max = float(cfg["sfincs"]["boundary_setup"]["river_fall_max_hr"])
    rise_max = 24.0 * float(
        cfg["boundary_forcings"]["river"]["event_hydrograph"]["max_rising_days"]
    )
    h = ILLUSTRATIVE_HR
    lag = h["surge_travel"] - h["river_travel"]
    river_lead = h["river_rise"] + h["river_ramp"]
    river_tail = h["river_fall"] + h["river_travel"]
    surge_hr, river_hr, end_hr = event_timeline(
        0.0,
        surge_ramp,
        window,
        h["surge_travel"],
        river_lead,
        river_tail,
        lag,
        surge_event=True,
        river_event=True,
    )
    t = np.linspace(0.0, end_hr, 1200)
    blend_start = surge_hr - window - surge_ramp
    tide = 0.12 * np.cos(2.0 * np.pi * (t - surge_hr) / 12.42)
    surge = 0.8 * np.exp(-0.5 * ((t - surge_hr) / 9.0) ** 2)
    sea = 0.3 + cosine_ramp(t, blend_start, surge_ramp) * (tide + surge)
    ramp_start = river_hr - river_lead
    q_start = 0.45
    rise = q_start + (1.0 - q_start) * 0.5 * (
        1.0 + np.cos(np.pi * np.clip((river_hr - t) / h["river_rise"], 0, 1))
    )
    fall = q_start + (1.0 - q_start) * 0.5 * (
        1.0 + np.cos(np.pi * np.clip((t - river_hr) / h["river_fall"], 0, 1))
    )
    river = np.where(t <= river_hr, rise, fall)
    river = np.where(
        t < river_hr - h["river_rise"],
        0.15 + (q_start - 0.15) * cosine_ramp(t, ramp_start, h["river_ramp"]),
        river,
    )
    river = np.where(
        t > river_hr + h["river_fall"],
        q_start - 0.1 * cosine_ramp(t, river_hr + h["river_fall"], 20.0),
        river,
    )
    _traces(ax, t, sea, river, "Hours since the restart (end of the spin-up)")
    for x, y0 in ((surge_hr, SEA_Y0 + 1.25), (river_hr, RIVER_Y0 + 1.05)):
        ax.plot([x], [y0 - 0.05], marker="v", color="0.2", markersize=4)
    ax.annotate(
        "",
        xy=(river_hr, RIVER_Y0 + 1.2),
        xytext=(surge_hr, RIVER_Y0 + 1.2),
        arrowprops=dict(arrowstyle="<->", lw=0.6, color="0.25", shrinkA=0, shrinkB=0),
    )
    ax.text(
        0.5 * (surge_hr + river_hr),
        RIVER_Y0 + 1.27,
        "lag = surge - river travel time:\npeaks meet at the main mouth",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    ax.axvline(end_hr, color="0.6", lw=0.6, linestyle="--")
    # sea row
    y = SEA_Y0 - 0.3
    _bracket(
        ax,
        blend_start,
        blend_start + surge_ramp,
        y,
        f"ramp\n{surge_ramp:g} h",
        ha="right",
    )
    _bracket(ax, blend_start + surge_ramp, surge_hr, y, f"window_hr {window:g} h")
    _bracket(ax, surge_hr, surge_hr + window, y, f"window_hr {window:g} h")
    _bracket(ax, surge_hr + window, end_hr, y, "surge\ntravel", ha="right")
    # river row
    y = RIVER_Y0 - 0.25
    _bracket(ax, ramp_start, river_hr - h["river_rise"], y, "ramp from\nthe mean")
    _bracket(
        ax,
        river_hr - h["river_rise"],
        river_hr,
        y,
        f"from Q_start\n(<= {rise_max:g} h)",
    )
    _bracket(
        ax, river_hr, river_hr + h["river_fall"], y, f"to Q_start\n(<= {fall_max:g} h)"
    )
    _bracket(ax, river_hr + h["river_fall"], river_hr + river_tail, y, "river\ntravel")
    ax.text(
        end_hr,
        -1.25,
        "end = the later of the two parts; the earlier-starting part begins at the restart",
        ha="right",
        va="top",
        fontsize=7,
    )
    ax.set_ylim(-1.55, RIVER_Y0 + 2.0)


def main() -> int:
    script_parser(__doc__).parse_args()
    apply_style()
    cfg = _config()
    fig, axes = plt.subplots(
        3, 1, figsize=figure_size(215.0), height_ratios=[1.0, 1.0, 1.25]
    )
    calibration_panel(axes[0], cfg)
    spinup_panel(axes[1], cfg)
    event_panel(axes[2], cfg)
    panel_letters(axes, xy=(-0.12, 1.0))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.05, hspace=0.42)
    path = analysis_path(None, __file__, "schematic.png")
    save_figure(fig, path)
    plt.close(fig)
    write_caption(
        path,
        "Each SFINCS run gets its length from a different rule: the calibration "
        "from the river's travel or settling time, the spin-up from a steady-state "
        "test, the event run from the event's own shape and travel times.\n"
        "(a) Rule 11 crest calibration (rounds 1-2): sea and rivers half-cosine "
        "ramped over ramp_hr, then held; run = ramp_hr + max(T_walled, settle_factor x "
        "t_settle) + calibration_margin_hr, with T_walled the slowest walled travel "
        "time and t_settle = L^2/D the settle time from a crossing to the network end.\n"
        "(b) Rule 13a spin-up (per slr_m): constant mean discharge and calm sea plus "
        "SLR x fingerprint, the boundary ramped from the sea's starting level over "
        "boundary_ramp_hr; run in chunks of chunk_hr until the river network passes the "
        "steady check over the chunk's last window_hr (grey); its end is the events' "
        "restart time.\n"
        "(c) Rule 14 event run (compound, lag_hr auto; src.sfincs_run.event_timeline): "
        "surge part = blend ramp + window_hr before and after the peak + surge travel "
        "to the main mouth; river part = ramp from the mean to Q_start + rise from "
        "Q_start (<= max_rising_days) + fall to Q_start (<= river_fall_max_hr) + river "
        "travel. The peaks are offset by the lag so both reach the main mouth together "
        "(triangles); the part needing more time before its peak starts at the restart.\n"
        "Blue solid line, sea level; orange dashed line, river discharge; both "
        "normalised (no scale). Fixed lengths are the config.yml values; travel, "
        "settle and river event times and the number of spin-up chunks are "
        "illustrative and differ per basin.",
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
