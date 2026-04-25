from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parent.parent
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]
BG = "#fbfaf7"
FG = "#18212b"
GRID = "#d7d3ca"
ACCENT = "#2f6fed"
ACCENT_ALT = "#d97706"
ACCENT_SOFT = "#7a3cff"
ACCENT_RED = "#c65b4b"
MUTED = "#6b7280"


@dataclass(frozen=True)
class RunOutputs:
    label: str
    y_true: np.ndarray
    probs: np.ndarray
    classes: list[str]
    metrics: dict


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_run_outputs(run_dir: Path, label: str) -> RunOutputs:
    obj = np.load(run_dir / "test_outputs.npz", allow_pickle=True)
    metrics = _load_json(run_dir / "metrics.json")
    return RunOutputs(
        label=label,
        y_true=np.asarray(obj["y_true"], dtype=np.int64),
        probs=np.asarray(obj["probs"], dtype=np.float64),
        classes=[str(x) for x in obj["classes"].tolist()],
        metrics=metrics,
    )


def _load_metadata(metadata_csv: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            rows[image_id] = row
    return rows


def _age_bucket(age_text: str) -> str:
    try:
        age = float(age_text)
    except (TypeError, ValueError):
        return "unknown"
    if age < 40:
        return "<40"
    if age < 60:
        return "40-59"
    if age < 80:
        return "60-79"
    return "80+"


def _normalize_group(value: str | None) -> str:
    if value is None:
        return "unknown"
    value = value.strip().lower()
    return value if value else "unknown"


def _top1_predictions(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.argmax(probs, axis=1)
    confidence = probs[np.arange(len(probs)), pred]
    return pred, confidence


def _calibration_stats(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> dict:
    pred, confidence = _top1_predictions(probs)
    correct = pred == y_true
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    points: list[dict[str, float]] = []
    ece = 0.0

    for idx in range(bins):
        lo = bin_edges[idx]
        hi = bin_edges[idx + 1]
        if idx == bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            points.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "center": float((lo + hi) / 2.0),
                    "accuracy": 0.0,
                    "avg_conf": 0.0,
                    "count": 0,
                }
            )
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(confidence[mask].mean())
        ece += (count / max(1, len(y_true))) * abs(acc - avg_conf)
        points.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "center": float((lo + hi) / 2.0),
                "accuracy": acc,
                "avg_conf": avg_conf,
                "count": count,
            }
        )

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0))))
    return {
        "ece": float(ece),
        "brier": brier,
        "nll": nll,
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "points": points,
        "correct": correct.astype(np.int64),
        "confidence": confidence,
    }


def _subgroup_recall(
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
    groups: list[str],
) -> dict[str, dict[str, float]]:
    mel_idx = classes.index("mel")
    pred = np.argmax(probs, axis=1)
    stats: dict[str, dict[str, float]] = {}
    unique_groups = list(dict.fromkeys(groups))
    for group in unique_groups:
        mask = np.array([(g == group) for g in groups], dtype=bool)
        mel_mask = mask & (y_true == mel_idx)
        count = int(mel_mask.sum())
        if count == 0:
            recall = 0.0
        else:
            recall = float((pred[mel_mask] == mel_idx).mean())
        stats[group] = {"mel_count": count, "mel_recall": recall}
    return stats


def _with_default_groups(
    stats: dict[str, dict[str, float]],
    group_order: list[str],
) -> dict[str, dict[str, float]]:
    out = dict(stats)
    for group in group_order:
        out.setdefault(group, {"mel_count": 0, "mel_recall": 0.0})
    return out


def _safe_float(value: float) -> str:
    return f"{value:.3f}"


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str) -> None:
    w, h = _text_size(draw, text, font)
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=font, fill=fill)


def _new_canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1200, 760), BG)
    draw = ImageDraw.Draw(image)
    title_font = _load_font(34)
    subtitle_font = _load_font(18)
    draw.text((64, 36), title, font=title_font, fill=FG)
    draw.text((64, 82), subtitle, font=subtitle_font, fill=MUTED)
    return image, draw


def _axes_frame(draw: ImageDraw.ImageDraw, left: int = 90, top: int = 150, right: int = 1120, bottom: int = 660) -> tuple[int, int, int, int]:
    draw.rectangle((left, top, right, bottom), outline=GRID, width=2)
    return left, top, right, bottom


def _plot_area(x: float, y: float, bounds: tuple[int, int, int, int]) -> tuple[float, float]:
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    px = left + x * width
    py = bottom - y * height
    return px, py


def _draw_y_grid(draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int], ticks: int = 5) -> None:
    left, top, right, bottom = bounds
    font = _load_font(16)
    for step in range(ticks + 1):
        value = step / ticks
        _, py = _plot_area(0.0, value, bounds)
        draw.line((left, py, right, py), fill=GRID, width=1)
        label = f"{value:.1f}"
        draw.text((left - 46, py - 8), label, font=font, fill=MUTED)


def _draw_x_ticks(draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int], labels: list[str]) -> None:
    font = _load_font(15)
    n = max(1, len(labels))
    for idx, label in enumerate(labels):
        x = (idx + 0.5) / n
        px, py = _plot_area(x, 0.0, bounds)
        draw.line((px, py, px, py + 8), fill=FG, width=2)
        tw, _ = _text_size(draw, label, font)
        draw.text((px - tw / 2, py + 16), label, font=font, fill=MUTED)


def _draw_legend(draw: ImageDraw.ImageDraw, items: list[tuple[str, str]], x: int | None = None, y: int = 42) -> None:
    font = _load_font(16)
    box = 16
    total_width = 0
    for label, _ in items:
        total_width += box + 8 + _text_size(draw, label, font)[0] + 30
    total_width = max(0, total_width - 30)
    if x is None:
        x = max(420, 1200 - total_width - 40)
    current_x = x
    for label, color in items:
        draw.rounded_rectangle((current_x, y, current_x + box, y + box), radius=3, fill=color)
        draw.text((current_x + box + 8, y - 2), label, font=font, fill=FG)
        current_x += box + 8 + _text_size(draw, label, font)[0] + 30


def _save_reliability_plot(stats: dict, out_path: Path, label: str) -> None:
    image, draw = _new_canvas(
        "Reliability Diagram",
        f"{label}: top-1 confidence versus observed accuracy across 10 bins",
    )
    bounds = _axes_frame(draw)
    _draw_y_grid(draw, bounds)

    label_font = _load_font(18)
    draw.text((515, 690), "Confidence bin", font=label_font, fill=FG)
    draw.text((18, 396), "Accuracy", font=label_font, fill=FG)

    left, top, right, bottom = bounds
    draw.line((_plot_area(0.0, 0.0, bounds), _plot_area(1.0, 1.0, bounds)), fill=MUTED, width=2)

    points = stats["points"]
    n = len(points)
    for idx, point in enumerate(points):
        x0 = left + (idx / n) * (right - left)
        x1 = left + ((idx + 1) / n) * (right - left)
        bar_w = max(10, int((x1 - x0) * 0.62))
        bar_x = x0 + ((x1 - x0) - bar_w) / 2
        acc_y = _plot_area(0.0, point["accuracy"], bounds)[1]
        draw.rounded_rectangle((bar_x, acc_y, bar_x + bar_w, bottom), radius=5, fill=ACCENT)

        conf_x = (bar_x + bar_x + bar_w) / 2
        conf_y = _plot_area(0.0, point["avg_conf"], bounds)[1]
        draw.ellipse((conf_x - 5, conf_y - 5, conf_x + 5, conf_y + 5), fill=ACCENT_ALT)
        draw.line((conf_x, conf_y - 10, conf_x, conf_y + 10), fill=ACCENT_ALT, width=2)

        tick_label = f"{point['lo']:.1f}-{point['hi']:.1f}"
        tw, _ = _text_size(draw, tick_label, _load_font(14))
        draw.text((x0 + (x1 - x0 - tw) / 2, bottom + 18), tick_label, font=_load_font(14), fill=MUTED)

    _draw_legend(draw, [("Observed accuracy", ACCENT), ("Average confidence", ACCENT_ALT)])

    note_font = _load_font(18)
    stats_box = (
        f"Accuracy {stats['accuracy']:.3f}   "
        f"Mean confidence {stats['mean_confidence']:.3f}   "
        f"ECE {stats['ece']:.3f}   "
        f"Brier {stats['brier']:.3f}"
    )
    draw.rounded_rectangle((64, 110, 840, 138), radius=8, fill="#eee9df")
    draw.text((78, 114), stats_box, font=note_font, fill=FG)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _save_confidence_histogram(stats: dict, out_path: Path, label: str) -> None:
    image, draw = _new_canvas(
        "Confidence Histogram",
        f"{label}: prediction confidence separated by correct and incorrect predictions",
    )
    bounds = _axes_frame(draw)
    label_font = _load_font(18)
    draw.text((518, 690), "Confidence bin", font=label_font, fill=FG)
    draw.text((28, 396), "Count", font=label_font, fill=FG)

    conf = np.asarray(stats["confidence"])
    correct = np.asarray(stats["correct"], dtype=bool)
    bins = np.linspace(0.0, 1.0, 11)
    correct_counts, _ = np.histogram(conf[correct], bins=bins)
    wrong_counts, _ = np.histogram(conf[~correct], bins=bins)
    max_count = max(1, int(max(correct_counts.max(initial=0), wrong_counts.max(initial=0))))

    left, top, right, bottom = bounds
    font14 = _load_font(14)
    for step in range(6):
        value = step / 5
        y = int(bottom - value * (bottom - top))
        draw.line((left, y, right, y), fill=GRID, width=1)
        tick_value = int(round(value * max_count))
        draw.text((left - 46, y - 8), str(tick_value), font=font14, fill=MUTED)

    n = 10
    for idx in range(n):
        x0 = left + idx * (right - left) / n
        x1 = left + (idx + 1) * (right - left) / n
        mid = (x0 + x1) / 2
        pair_width = (x1 - x0) * 0.64
        bar_w = pair_width / 2 - 6

        for bar_idx, (count, color) in enumerate(
            [(int(correct_counts[idx]), ACCENT), (int(wrong_counts[idx]), ACCENT_RED)]
        ):
            height = 0 if max_count == 0 else (count / max_count) * (bottom - top)
            bx0 = mid - pair_width / 2 + bar_idx * (bar_w + 12)
            bx1 = bx0 + bar_w
            by0 = bottom - height
            draw.rounded_rectangle((bx0, by0, bx1, bottom), radius=5, fill=color)

        tick_label = f"{bins[idx]:.1f}-{bins[idx+1]:.1f}"
        tw, _ = _text_size(draw, tick_label, font14)
        draw.text((x0 + (x1 - x0 - tw) / 2, bottom + 18), tick_label, font=font14, fill=MUTED)

    _draw_legend(draw, [("Correct", ACCENT), ("Incorrect", ACCENT_RED)])
    draw.rounded_rectangle((64, 110, 634, 138), radius=8, fill="#eee9df")
    draw.text(
        (78, 114),
        f"High-confidence errors cluster near the right edge if the model is overconfident. ECE = {stats['ece']:.3f}",
        font=_load_font(18),
        fill=FG,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _save_grouped_recall_chart(
    *,
    out_path: Path,
    title: str,
    subtitle: str,
    group_order: list[str],
    group_counts: dict[str, int],
    series: list[tuple[str, dict[str, dict[str, float]], str]],
) -> None:
    image, draw = _new_canvas(title, subtitle)
    bounds = _axes_frame(draw)
    _draw_y_grid(draw, bounds)
    left, top, right, bottom = bounds
    font14 = _load_font(14)
    label_font = _load_font(18)
    draw.text((528, 712), "Subgroup", font=label_font, fill=FG)
    draw.text((24, 392), "Melanoma recall", font=label_font, fill=FG)
    _draw_legend(draw, [(label, color) for label, _, color in series])

    n_groups = len(group_order)
    n_series = len(series)
    plot_w = right - left
    group_w = plot_w / max(1, n_groups)
    inner_w = group_w * 0.7
    bar_w = max(22, int((inner_w - 18) / max(1, n_series)))

    for group_idx, group in enumerate(group_order):
        gx0 = left + group_idx * group_w + (group_w - inner_w) / 2
        for series_idx, (_, values, color) in enumerate(series):
            recall = values[group]["mel_recall"]
            bx0 = gx0 + series_idx * bar_w + series_idx * 8
            bx1 = bx0 + bar_w
            by0 = _plot_area(0.0, recall, bounds)[1]
            draw.rounded_rectangle((bx0, by0, bx1, bottom), radius=5, fill=color)
            label = _safe_float(recall)
            tw, th = _text_size(draw, label, font14)
            draw.text((bx0 + (bar_w - tw) / 2, by0 - th - 6), label, font=font14, fill=FG)

        center_x = gx0 + inner_w / 2
        group_label = f"{group}\n(n={group_counts[group]})"
        tw, _ = _text_size(draw, group, font14)
        draw.text((center_x - tw / 2, bottom + 16), group, font=font14, fill=MUTED)
        count_text = f"n={group_counts[group]}"
        tw2, _ = _text_size(draw, count_text, font14)
        draw.text((center_x - tw2 / 2, bottom + 36), count_text, font=font14, fill=MUTED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _write_report(
    *,
    out_path: Path,
    main_run: RunOutputs,
    compare_run: RunOutputs,
    calibration: dict,
    sex_stats_main: dict[str, dict[str, float]],
    sex_stats_compare: dict[str, dict[str, float]],
    age_stats_main: dict[str, dict[str, float]],
    age_stats_compare: dict[str, dict[str, float]],
) -> None:
    lines: list[str] = []
    lines.append("# Advanced Post-hoc Analysis\n")
    lines.append("This add-on analysis keeps the original training results unchanged and adds confidence + subgroup diagnostics.\n")

    lines.append("## Calibration snapshot\n")
    lines.append(f"- Model: **{main_run.label}**")
    lines.append(f"- Test accuracy: **{main_run.metrics['test']['accuracy']:.4f}**")
    lines.append(f"- Mean top-1 confidence: **{calibration['mean_confidence']:.4f}**")
    lines.append(f"- Expected calibration error (ECE): **{calibration['ece']:.4f}**")
    lines.append(f"- Multi-class Brier score: **{calibration['brier']:.4f}**")
    lines.append(f"- Negative log-likelihood: **{calibration['nll']:.4f}**\n")

    lines.append("## Melanoma recall trade-off\n")
    lines.append(f"- {main_run.label}: accuracy **{main_run.metrics['test']['accuracy']:.4f}**, melanoma recall **{main_run.metrics['test']['per_class_recall']['mel']:.4f}**")
    lines.append(f"- {compare_run.label}: accuracy **{compare_run.metrics['test']['accuracy']:.4f}**, melanoma recall **{compare_run.metrics['test']['per_class_recall']['mel']:.4f}**\n")

    lines.append("## Melanoma recall by sex\n")
    for group in sex_stats_main:
        lines.append(
            f"- `{group}` (n={sex_stats_main[group]['mel_count']} melanoma cases): "
            f"{main_run.label}={sex_stats_main[group]['mel_recall']:.3f}, "
            f"{compare_run.label}={sex_stats_compare[group]['mel_recall']:.3f}"
        )
    lines.append("")

    lines.append("## Melanoma recall by age bucket\n")
    for group in age_stats_main:
        lines.append(
            f"- `{group}` (n={age_stats_main[group]['mel_count']} melanoma cases): "
            f"{main_run.label}={age_stats_main[group]['mel_recall']:.3f}, "
            f"{compare_run.label}={age_stats_compare[group]['mel_recall']:.3f}"
        )
    lines.append("")

    lines.append("## Generated figures\n")
    lines.append("- `advanced/reliability_effnetb2_260_acc.png`")
    lines.append("- `advanced/confidence_hist_effnetb2_260_acc.png`")
    lines.append("- `advanced/mel_recall_by_sex_compare.png`")
    lines.append("- `advanced/mel_recall_by_age_compare.png`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-run", type=Path, required=True)
    parser.add_argument("--compare-run", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=REPO_ROOT / "data" / "ham10000" / "HAM10000_metadata.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "advanced")
    parser.add_argument("--main-label", type=str, default="EffNet-B2 260px (accuracy-focused)")
    parser.add_argument("--compare-label", type=str, default="EffNet-B2 260px (sensitivity-first)")
    args = parser.parse_args()

    main_run = _load_run_outputs(args.main_run, args.main_label)
    compare_run = _load_run_outputs(args.compare_run, args.compare_label)
    if main_run.classes != compare_run.classes:
        raise ValueError("Class order mismatch between runs.")
    if not np.array_equal(main_run.y_true, compare_run.y_true):
        raise ValueError("Test labels mismatch between runs.")

    split = _load_json(args.main_run / "split.json")
    image_ids = split["test_image_ids"]
    if len(image_ids) != len(main_run.y_true):
        raise ValueError("split.json test_image_ids length does not match test outputs.")

    metadata = _load_metadata(args.metadata_csv)
    sex_groups = []
    age_groups = []
    for image_id in image_ids:
        row = metadata.get(image_id)
        if row is None:
            sex_groups.append("unknown")
            age_groups.append("unknown")
            continue
        sex_groups.append(_normalize_group(row.get("sex")))
        age_groups.append(_age_bucket(row.get("age", "")))

    calibration = _calibration_stats(main_run.y_true, main_run.probs)
    sex_stats_main = _subgroup_recall(y_true=main_run.y_true, probs=main_run.probs, classes=main_run.classes, groups=sex_groups)
    sex_stats_compare = _subgroup_recall(
        y_true=compare_run.y_true,
        probs=compare_run.probs,
        classes=compare_run.classes,
        groups=sex_groups,
    )
    age_order = ["<40", "40-59", "60-79", "80+", "unknown"]
    sex_order = ["female", "male", "unknown"]
    age_stats_main = _subgroup_recall(y_true=main_run.y_true, probs=main_run.probs, classes=main_run.classes, groups=age_groups)
    age_stats_compare = _subgroup_recall(
        y_true=compare_run.y_true,
        probs=compare_run.probs,
        classes=compare_run.classes,
        groups=age_groups,
    )
    sex_stats_main = _with_default_groups(sex_stats_main, sex_order)
    sex_stats_compare = _with_default_groups(sex_stats_compare, sex_order)
    age_stats_main = _with_default_groups(age_stats_main, age_order)
    age_stats_compare = _with_default_groups(age_stats_compare, age_order)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_reliability_plot(calibration, out_dir / "reliability_effnetb2_260_acc.png", args.main_label)
    _save_confidence_histogram(calibration, out_dir / "confidence_hist_effnetb2_260_acc.png", args.main_label)
    _save_grouped_recall_chart(
        out_path=out_dir / "mel_recall_by_sex_compare.png",
        title="Melanoma Recall by Sex",
        subtitle="Argmax melanoma recall on the shared test split; counts refer to melanoma cases in each subgroup",
        group_order=sex_order,
        group_counts={g: sex_stats_main[g]["mel_count"] for g in sex_order},
        series=[
            ("Accuracy-focused", sex_stats_main, ACCENT),
            ("Sensitivity-first", sex_stats_compare, ACCENT_SOFT),
        ],
    )
    _save_grouped_recall_chart(
        out_path=out_dir / "mel_recall_by_age_compare.png",
        title="Melanoma Recall by Age Bucket",
        subtitle="Argmax melanoma recall on the shared test split; counts refer to melanoma cases in each subgroup",
        group_order=age_order,
        group_counts={g: age_stats_main[g]["mel_count"] for g in age_order},
        series=[
            ("Accuracy-focused", age_stats_main, ACCENT),
            ("Sensitivity-first", age_stats_compare, ACCENT_SOFT),
        ],
    )
    _write_report(
        out_path=out_dir / "README.md",
        main_run=main_run,
        compare_run=compare_run,
        calibration=calibration,
        sex_stats_main=sex_stats_main,
        sex_stats_compare=sex_stats_compare,
        age_stats_main=age_stats_main,
        age_stats_compare=age_stats_compare,
    )
    print(f"Wrote analysis bundle to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
