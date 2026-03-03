from __future__ import annotations

import argparse
import math
import subprocess  # noqa: S404

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .cli import annotate_frame_candidates, iterm2_display
from .lib import (
    Band,
    build_shots,
    detect_band,
    find_band_candidates,
    get_duration_seconds,
    repair_band_inpaint,
    run_ffmpeg_scene_detect,
)


def main() -> None:  # noqa: C901, PLR0912, PLR0914, PLR0915
    ap = argparse.ArgumentParser(
        prog="fix-video",
        description="Detect and repair stuck-pixel vertical bands in video files.",
    )
    ap.add_argument("input", help="Input video")
    ap.add_argument("output", help="Output video")
    ap.add_argument("--tmp-video", default="__fixed_video.mp4", help="Intermediate video-only path")
    ap.add_argument("--scene-thresh", type=float, default=0.30, help="FFmpeg scene threshold (higher = fewer cuts)")
    ap.add_argument("--min-shot-len", type=float, default=0.50, help="Minimum shot length seconds (merge shorter)")
    ap.add_argument("--samples", type=int, default=7, help="Frames sampled per shot for band estimation")
    ap.add_argument("--detect-win", type=int, default=31, help="Detection rolling median window (odd)")
    ap.add_argument("--detect-z", type=float, default=6.0, help="Detection threshold in robust sigmas")
    ap.add_argument("--band-width", type=int, default=4, help="Exact defect band width in pixels")
    ap.add_argument("--auto-thresh", type=float, default=0.7,
                    help="Auto-select best candidate when runner-up scores < this fraction of the winner (0-1)")
    ap.add_argument("--pad", type=int, default=3, help="Repair pad: inpaint radius outside band mask")
    ap.add_argument("--codec", default="avc1", help="FourCC for intermediate (avc1 is H.264)")
    ap.add_argument("--display-cols", type=int, default=120,
                    help="Width of inline frame previews in terminal columns (iTerm2)")
    args = ap.parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    console = Console()
    console.print(Panel.fit(
        f"[bold cyan]fix_video[/]\n"
        f"[dim]Input :[/]  {args.input}\n"
        f"[dim]Output:[/]  {args.output}",
        border_style="cyan",
    ))

    # ── 1 / 3  Scene Detection ────────────────────────────────────────────────
    console.rule("[bold cyan]1 / 3  Scene Detection[/]")
    with console.status(f"[cyan]Running ffmpeg scene detect  (thresh={args.scene_thresh})…[/]"):
        cut_times = run_ffmpeg_scene_detect(args.input, args.scene_thresh)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        console.print(f"[bold red]ERROR:[/] Failed to open: {args.input}")
        raise SystemExit(1)

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = get_duration_seconds(cap)

    console.print(
        f"  [dim]Duration:[/] [bold]{duration:.2f}s[/]  "
        f"[dim]FPS:[/] [bold]{fps:.3f}[/]  "
        f"[dim]Size:[/] [bold]{w}*{h}[/]  "
        f"[dim]Frames:[/] [bold]{total_frame_count}[/]",
    )

    shots = build_shots(cut_times, duration=duration, min_shot_len=args.min_shot_len)
    console.print(
        f"  [dim]Cuts found:[/] [bold]{len(cut_times)}[/]  →  "
        f"[dim]Shots:[/] [bold]{len(shots)}[/]",
    )

    if not shots:
        console.print(
            "[bold red]ERROR:[/] No shots found. "
            "Adjust [dim]--scene-thresh[/] or [dim]--min-shot-len[/].",
        )
        cap.release()
        raise SystemExit(1)

    # ── 2 / 3  Band Estimation ────────────────────────────────────────────────
    console.rule("[bold cyan]2 / 3  Band Estimation[/]")

    shot_bands: list[Band] = []

    for i, shot in enumerate(shots):
        shot_dur = shot.t1 - shot.t0
        console.print(
            f"\n  [bold]Shot {i + 1}/{len(shots)}[/]  "
            f"[dim]{shot.t0:.2f}s → {shot.t1:.2f}s[/]  "
            f"({shot_dur:.2f}s)",
        )

        # Build sample timestamps
        eps = min(0.2, shot_dur * 0.1)
        t0s = shot.t0 + eps
        t1s = shot.t1 - eps
        if t1s <= t0s:
            t0s, t1s = shot.t0, shot.t1
        n = args.samples
        times: list[float] = [t0s] if n == 1 else list(np.linspace(t0s, t1s, n))

        col_means: list[np.ndarray] = []  # accumulated per-frame BGR column profiles (W, 3)
        last_read_frame: np.ndarray | None = None   # any successfully read frame
        last_good_frame: np.ndarray | None = None   # last frame with a detected band
        per_frame_detections: int = 0

        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                console.print(f"    [dim]t={t:8.3f}s[/]  [yellow]⚠  could not read frame[/]")
                continue
            last_read_frame = frame

            col_means.append(frame.astype(np.float32).mean(axis=0))  # (W, 3)

            # Per-frame detection (debug display only; aggregate is authoritative)
            band = detect_band(
                col_means[-1],
                win=args.detect_win,
                z_thresh=args.detect_z,
                band_width=args.band_width,
            )

            if band is not None:
                per_frame_detections += 1
                last_good_frame = frame
                console.print(
                    f"    [dim]t={t:8.3f}s[/]  "
                    f"[green]✓[/]  band x=[{band.x0}..{band.x1}]",
                )
            else:
                console.print(f"    [dim]t={t:8.3f}s[/]  [yellow]✗  no band[/]")

        if not col_means:
            console.print(
                f"\n  [bold red]ERROR: Shot {i + 1} — could not read any frames![/]",
            )
            cap.release()
            raise SystemExit(1)

        # ── Temporal-aggregate candidate detection ────────────────────────────
        agg_profile = np.median(np.stack(col_means), axis=0)
        candidates = find_band_candidates(
            agg_profile,
            win=args.detect_win,
            z_thresh=args.detect_z,
            band_width=args.band_width,
        )

        if not candidates:
            console.print(
                f"\n  [bold red]ERROR: Shot {i + 1} — no candidates found in temporal "
                f"aggregate of {len(col_means)} frames.[/]\n"
                f"  [dim]Per-frame detections: {per_frame_detections}/{len(col_means)}\n"
                f"  Hints: lower [bold]--detect-z[/] (current {args.detect_z}), "
                f"adjust [bold]--band-width[/] (current {args.band_width}), "
                f"or raise [bold]--samples[/].[/]",
            )
            cap.release()
            raise SystemExit(1)

        # Display all candidates on the reference frame
        ref_frame = last_good_frame if last_good_frame is not None else last_read_frame
        assert ref_frame is not None  # noqa: S101  # col_means non-empty guarantees at least one read
        iterm2_display(ref_frame, cols=args.display_cols)
        print()  # noqa: T201
        preview = annotate_frame_candidates(
            ref_frame, candidates, selected_idx=0,
            label=f"Shot {i + 1} / {len(shots)}  —  {len(candidates)} candidate(s)  "
                  f"[aggregate, {per_frame_detections}/{len(col_means)} per-frame hits]",
        )
        iterm2_display(preview, cols=args.display_cols)
        print()  # noqa: T201

        # Print ranked candidate list
        console.print(f"\n  [bold]Candidates[/]  (aggregate of {len(col_means)} frames):")
        for k, (band, score) in enumerate(candidates):
            tag = "  [bold green]← best[/]" if k == 0 else ""
            console.print(
                f"    [{'bold green' if k == 0 else 'dim'}][{k + 1}][/]"
                f"  x=[{band.x0}..{band.x1}]  score={score:.3f}{tag}",
            )

        # Auto-select if runner-up is not within auto_thresh of the winner
        best_score = candidates[0][1]
        runner_up_score = candidates[1][1] if len(candidates) > 1 else 0.0
        auto_select = runner_up_score < args.auto_thresh * best_score

        if auto_select:
            estimated = candidates[0][0]
            console.print(
                f"  [bold green]→ Auto-selected:[/] x=[{estimated.x0}..{estimated.x1}]  "
                f"[dim](runner-up {runner_up_score:.3f} < {args.auto_thresh:.0%} * {best_score:.3f})[/]",
            )
            shot_bands.append(estimated)
        else:
            console.print(
                f"  [yellow]Runner-up score {runner_up_score:.3f} is within "
                f"{args.auto_thresh:.0%} of winner {best_score:.3f} — manual selection required.[/]",
            )
            estimated = None
            while estimated is None:
                try:
                    raw = input(f"\n  Select [1]-[{len(candidates)}], or enter x0 (Enter = 1): ").strip()
                except EOFError:
                    raw = ""
                if not raw:
                    estimated = candidates[0][0]
                elif raw.isdigit():
                    n = int(raw)
                    if 1 <= n <= len(candidates):
                        estimated = candidates[n - 1][0]
                    else:
                        print(f"  Enter a number between 1 and {len(candidates)}.")  # noqa: T201
                else:
                    try:
                        x0 = int(raw)
                        estimated = Band(x0, x0 + args.band_width - 1)
                    except ValueError:
                        print("  Enter a candidate number or a pixel column x0.")  # noqa: T201
            console.print(
                f"  [bold green]→ Selected:[/] x=[{estimated.x0}..{estimated.x1}]",
            )
            shot_bands.append(estimated)

    # Reset for sequential read
    cap.release()
    cap = cv2.VideoCapture(args.input)

    fourcc = cv2.VideoWriter.fourcc(*args.codec)
    vw = cv2.VideoWriter(args.tmp_video, fourcc, fps, (w, h))
    if not vw.isOpened():
        console.print(
            "[bold red]ERROR:[/] Failed to open VideoWriter. "
            "Try [dim]--codec mp4v[/] and/or a [dim].mp4[/] output path.",
        )
        cap.release()
        raise SystemExit(1)

    # ── 3 / 3  Frame Processing ───────────────────────────────────────────────
    console.rule("[bold cyan]3 / 3  Frame Processing[/]")
    console.print(f"  Writing to [dim]{args.tmp_video}[/]…\n")

    shot_idx = 0
    fixed_frames = 0
    total_frames = 0

    current_shot = shots[shot_idx]
    current_band = shot_bands[shot_idx]
    shot_end_frame = math.floor(current_shot.t1 * fps + 1e-6)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing frames", total=total_frame_count or None)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if total_frames >= shot_end_frame and shot_idx + 1 < len(shots):
                shot_idx += 1
                current_shot = shots[shot_idx]
                current_band = shot_bands[shot_idx]
                shot_end_frame = math.floor(current_shot.t1 * fps + 1e-6)

            frame = repair_band_inpaint(frame, current_band, radius=args.pad)
            fixed_frames += 1

            vw.write(frame)
            total_frames += 1
            progress.advance(task)

    cap.release()
    vw.release()

    # ── Mux audio ─────────────────────────────────────────────────────────────
    with console.status("[cyan]Muxing audio…[/]"):
        cmd = [
            "ffmpeg", "-y",
            "-i", args.tmp_video,
            "-i", args.input,
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "copy",
            args.output,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603

    # ── Summary ───────────────────────────────────────────────────────────────
    table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    table.add_column("#", style="dim", justify="right")
    table.add_column("Range")
    table.add_column("Band x0..x1")
    table.add_column("Width", justify="right")

    for i, (shot, band) in enumerate(zip(shots, shot_bands, strict=True)):
        table.add_row(
            str(i + 1),
            f"{shot.t0:.2f}s → {shot.t1:.2f}s",
            f"[{band.x0}..{band.x1}]",
            str(band.x1 - band.x0 + 1),
        )

    console.print()
    console.print(Panel(
        table,
        title="[bold green]Done[/]",
        subtitle=f"[dim]{args.output}[/]",
        border_style="green",
    ))
    console.print(
        f"  [dim]Frames total:[/] [bold]{total_frames}[/]  "
        f"[dim]Fixed:[/] [bold]{fixed_frames}[/]  "
        f"[dim]Shots:[/] [bold]{len(shots)}[/]",
    )


if __name__ == "__main__":
    main()
