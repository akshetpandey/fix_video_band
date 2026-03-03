from __future__ import annotations

import base64
import os
import sys
from typing import TYPE_CHECKING, Literal

import cv2

if TYPE_CHECKING:
    import numpy as np

    from fix_video_band.lib import Band

_KITTY_CHUNK = 4096

Terminal = Literal['iterm2', 'kitty', 'unsupported']

_SUPPORTED_ITERM2_TERM_STRINGS = ['iTerm.app']
_SUPPORTED_KITTY_TERM_PROGRAM_STRINGS = ['WezTerm', 'ghostty', 'konsole']
_SUPPORTED_KITTY_TERM_STRINGS = ['xterm-kitty', 'xterm-ghostty', 'wezterm', 'konsole-direct', 'konsole-256color']


def detect_terminal() -> Terminal:
    """Return which inline-image protocol the current terminal supports."""
    if os.environ.get('TERM_PROGRAM') in _SUPPORTED_ITERM2_TERM_STRINGS:
        return 'iterm2'
    if (
        os.environ.get('TERM_PROGRAM') in _SUPPORTED_KITTY_TERM_PROGRAM_STRINGS
        or os.environ.get('TERM') in _SUPPORTED_KITTY_TERM_STRINGS
        or 'KITTY_WINDOW_ID' in os.environ
    ):
        return 'kitty'
    return 'unsupported'


def annotate_frame_candidates(
    frame_bgr: np.ndarray,
    candidates: list[tuple[Band, float]],
    selected_idx: int = 0,
    label: str = '',
    max_w: int = 960,
) -> np.ndarray:
    """
    Annotate a frame with all candidate bands.

    Each band is shown as a 1-px vertical line at the band's starting column (x0).
    The selected candidate is green; all others are yellow.  No fill overlay.
    """
    src_h, src_w = frame_bgr.shape[:2]
    scale = min(1.0, max_w / src_w)
    dw = int(src_w * scale)
    dh = int(src_h * scale)
    frame = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_AREA)

    font = cv2.FONT_HERSHEY_SIMPLEX

    def _text(img: np.ndarray, txt: str, x: int, y: int, fg: tuple[int, int, int]) -> None:
        cv2.putText(img, txt, (x, y), font, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (x, y), font, 0.52, fg, 1, cv2.LINE_AA)

    for k, (band, score) in enumerate(candidates):
        color: tuple[int, int, int] = (0, 255, 0) if k == selected_idx else (0, 220, 255)
        bx0 = max(0, int(band.x0 * scale))
        cv2.line(frame, (bx0, 0), (bx0, dh - 1), color, 1)
        lbl = f'[{k + 1}] x={band.x0}  {score:.2f}'
        tx = bx0 + 3 if bx0 + 110 < dw else max(0, bx0 - 108)
        _text(frame, lbl, tx, 18 + k * 20, fg=color)

    if label:
        (tw, th), _ = cv2.getTextSize(label, font, 0.48, 1)
        cv2.rectangle(frame, (4, dh - th - 12), (tw + 10, dh - 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (7, dh - 7), font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def _iterm2_display(frame_bgr: np.ndarray, cols: int = 120) -> None:
    """Print a BGR frame inline using the iTerm2 image protocol."""
    _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    data = buf.tobytes()
    b64 = base64.b64encode(data).decode()
    sys.stdout.write(
        f'\x1b]1337;File=inline=1;size={len(data)};width={cols};preserveAspectRatio=1:{b64}\a\n',
    )
    sys.stdout.flush()


def _kitty_display(frame_bgr: np.ndarray, cols: int = 120) -> None:
    """Print a BGR frame inline using the Kitty Graphics Protocol."""
    _, buf = cv2.imencode('.png', frame_bgr)
    b64 = base64.b64encode(buf.tobytes()).decode()
    chunks = [b64[i : i + _KITTY_CHUNK] for i in range(0, len(b64), _KITTY_CHUNK)]
    for i, chunk in enumerate(chunks):
        more = 0 if i == len(chunks) - 1 else 1
        header = f'a=T,f=100,q=2,c={cols},m={more}' if i == 0 else f'm={more}'
        sys.stdout.write(f'\x1b_G{header};{chunk}\x1b\\')
    sys.stdout.write('\n')
    sys.stdout.flush()


def display_frame(
    frame_bgr: np.ndarray,
    cols: int = 120,
    terminal: Terminal | None = None,
) -> None:
    """Display a BGR frame inline, auto-detecting the terminal protocol."""
    term = terminal if terminal is not None else detect_terminal()
    if term == 'iterm2':
        _iterm2_display(frame_bgr, cols)
    elif term == 'kitty':
        _kitty_display(frame_bgr, cols)
    else:
        print('image output unsupported')  # noqa: T201
