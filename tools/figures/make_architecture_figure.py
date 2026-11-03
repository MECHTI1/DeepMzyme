#!/usr/bin/env python3
"""Generate the README architecture figure (light and dark SVG variants).

Usage: python tools/figures/make_architecture_figure.py [--out-dir docs/assets]

Standard library only. The figure shows the public metal release
`unified_p5_trunklr1e4_5fold_s42` (src/metal_release/catalog.py): five copies of the
GVPPocketClassifier late-fusion network (src/model.py), each trained with one CV fold held
out, using the recipe in docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md. Every number comes from
that code path and recipe:

- pocket graph: protein residues with any atom within 10 A of the site (DEFAULT_POCKET_RADIUS);
  contacts link residues whose closest atoms are within 8 A (--edge-radius 8.0) and carry
  messages in both directions; the metal is not a graph node (--metal-node-mode none)
- first shell: a side-chain donor atom within 2.7 A of the metal (DEFAULT_FIRST_SHELL_CUTOFF);
  second shell: a side-chain group within 4.5 A of a first-shell group (shell_roles.py)
- a training pocket has 38 residues at the median (n_context_residues, train_cohort.csv)
- GVP-GNN: 4 message-passing layers; 128 scalar and 16 vector channels per residue
- ESM-C 600M: frozen; 1152 values per residue, each chain embedded in full
- pooling, both branches: mean and attention pooling over the pocket residues, concatenated
  (2 x 128) and projected to 128; the sequence branch first projects 1152 -> 128 per
  residue; the structure branch pools only the (rotation-invariant) scalar channels
- site summary: 4 values -> 32; constant in this release because each example holds one metal
- fusion: gate = sigmoid(W [h_struct ; h_seq]) scales h_seq element-wise;
  [h_struct ; gated h_seq ; h_site] = 128 + 128 + 32 = 288 values
- classifier: MLP 288 -> 128 -> 5 classes in the order Mn, Cu, Zn, Fe, Co/Ni; four-class
  scores sum Fe and Co/Ni into Class VIII

Update this file, not the generated SVGs, when the model changes. Every label carries a
data-fit box (the region it must stay inside); labels drawn on purpose over a glyph carry
data-overlay. A renderer can then check overflow, overlaps, contrast and lines through text.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.sax.saxutils import escape

RELEASE = "unified_p5_trunklr1e4_5fold_s42"
W, H = 1176, 806
SANS = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, "
        "sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji'")
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

# ink/ink2/txt3: text, each at least 4.5:1 against the fills it sits on (WCAG AA).
# ink3: lines and outlines only.
THEMES = {
    "light": {
        "ink": "#1f2328", "ink2": "#475467", "txt3": "#5b6576", "ink3": "#98a2b3", "rule": "#e4e7ec",
        "card": "#ffffff", "on": "#ffffff", "tile": "#ffffff",
        "s": "#0f766e", "s_fill": "#f0faf8", "s_soft": "#8fd6cb", "s_node": "#d5f1ec",
        "q": "#6d3fd6", "q_fill": "#f6f3ff", "q_frost": "#ebe5ff",
        "t": "#b45309", "t_fill": "#fff8ec", "t_glow": "#f59e0b",
        "f_fill": "#f6f8fa", "shadow": "#1f2328", "shadow_op": "0.07", "back": "#94a3b8",
    },
    "dark": {
        "ink": "#e6edf3", "ink2": "#b4bdc8", "txt3": "#8b949e", "ink3": "#5d6875", "rule": "#262c36",
        "card": "#0d1117", "on": "#0d1117", "tile": "#0d1117",
        "s": "#3cc9b4", "s_fill": "#0c2724", "s_soft": "#1f6b61", "s_node": "#123a35",
        "q": "#a98bff", "q_fill": "#1a1533", "q_frost": "#241d45",
        "t": "#f2b13c", "t_fill": "#271d0c", "t_glow": "#f59e0b",
        "f_fill": "#161b22", "shadow": "#000000", "shadow_op": "0.35", "back": "#8b95a3",
    },
}

# Column x-ranges and lane y-ranges.
COL = {"A": (24, 184), "B": (206, 466), "C": (486, 686), "D": (704, 834), "E": (844, 938), "F": (958, 1152)}
LANE_S = (60, 318)
LANE_Q = (368, 518)
LANE_T = (548, 664)
S_MID = (LANE_S[0] + LANE_S[1]) // 2
Q_MID = (LANE_Q[0] + LANE_Q[1]) // 2
T_MID = (LANE_T[0] + LANE_T[1]) // 2
POCKET = (104, 336, 74)  # centre x, centre y, radius of the 10 A sphere
BAR = (900, 920, 124, 1.27)  # concat bar x0, x1, top, pixels per channel
GATE_X, TAP_X = 872, 846
EXAMPLE_SEQUENCE = "KAHGEHVPHDLWTA"  # schematic fragment; highlighted letters stand for pocket residues
EXAMPLE_POCKET = {1, 2, 4, 5, 8, 9, 11}  # first shell H, E, H, H; second shell D; other pocket residues A, W
# Output classes in model order (element tiles).
CLASSES = [("Mn", ""), ("Cu", ""), ("Zn", ""), ("Fe", ""), ("Co/Ni", "")]

SIZE_TAG, SIZE_TITLE, SIZE_BODY = 11.5, 17, 13.5


class Svg:
    def __init__(self, p: dict[str, str]):
        self.p = p
        self.out: list[str] = []

    def add(self, markup: str) -> None:
        self.out.append(markup)

    def c(self, key: str) -> str:
        return self.p.get(key, key)

    def text(self, x, y, content, *, size=SIZE_BODY, weight=400, fill="ink", anchor="start", italic=False,
             spacing=None, fit=None, raw=False, overlay=False, mono=False):
        body = content if raw else escape(content)
        attrs = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'font-size="{size}"', f'fill="{self.c(fill)}"']
        if weight != 400:
            attrs.append(f'font-weight="{weight}"')
        if anchor != "start":
            attrs.append(f'text-anchor="{anchor}"')
        if italic:
            attrs.append('font-style="italic"')
        if mono:
            attrs.append('class="mono"')
        if spacing is not None:
            attrs.append(f'letter-spacing="{spacing}"')
        if fit is not None:
            attrs.append('data-fit="{:.1f},{:.1f},{:.1f},{:.1f}"'.format(*fit))
        if overlay:
            attrs.append('data-overlay="1"')
        self.add(f"<text {' '.join(attrs)}>{body}</text>")

    def rect(self, x0, y0, x1, y1, *, r=10, fill="card", stroke=None, width=1.2, dash=None, opacity=None,
             extra=""):
        attrs = [f'x="{x0:.1f}"', f'y="{y0:.1f}"', f'width="{x1 - x0:.1f}"', f'height="{y1 - y0:.1f}"',
                 f'rx="{r}"', f'fill="{self.c(fill)}"']
        if stroke:
            attrs.append(f'stroke="{self.c(stroke)}" stroke-width="{width}"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity}"')
        self.add(f"<rect {' '.join(attrs)} {extra}/>")

    def path(self, d, *, stroke="ink3", width=1.6, fill="none", dash=None, marker=None, opacity=None,
             cap="round", overlay=False):
        attrs = [f'd="{d}"', f'fill="{self.c(fill)}"', f'stroke="{self.c(stroke)}"', f'stroke-width="{width}"',
                 f'stroke-linecap="{cap}"', 'stroke-linejoin="round"']
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if marker:
            attrs.append(f'marker-end="url(#arrow-{marker})"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity}"')
        if overlay:
            attrs.append('data-overlay="1"')
        self.add(f"<path {' '.join(attrs)}/>")

    def circle(self, cx, cy, r, *, fill="card", stroke=None, width=1.2, opacity=None):
        attrs = [f'cx="{cx:.1f}"', f'cy="{cy:.1f}"', f'r="{r}"', f'fill="{self.c(fill)}"']
        if stroke:
            attrs.append(f'stroke="{self.c(stroke)}" stroke-width="{width}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity}"')
        self.add(f"<circle {' '.join(attrs)}/>")


def sub(base: str, subscript: str) -> str:
    """Italic base symbol with a lowered upright subscript (the text ends after the subscript)."""
    return (f'<tspan font-style="italic">{escape(base)}</tspan>'
            f'<tspan dy="4" font-size="70%" font-style="normal">{escape(subscript)}</tspan>')


def shadowed(g: Svg, x0, y0, x1, y1, *, r=12, fill="card", stroke="ink3", width=1.3, dash=None) -> None:
    g.rect(x0, y0 + 2.5, x1, y1 + 2.5, r=r, fill="shadow", opacity=g.c("shadow_op"))
    g.rect(x0, y0, x1, y1, r=r, fill=fill, stroke=stroke, width=width, dash=dash)


def card(g: Svg, col: str, lane: tuple[float, float], key: str, tag: str, title: str | None, *, fill=None,
         dash=None):
    """Card with a coloured tag (branch, plus role for encoders) and an optional title.

    Returns (fit box, body x, first body baseline)."""
    x0, x1 = COL[col]
    y0, y1 = lane
    shadowed(g, x0, y0, x1, y1, fill=fill or f"{key}_fill", stroke=key, dash=dash)
    fit = (x0 + 8, y0, x1 - 8, y1)
    g.text(x0 + 14, y0 + 22, tag, size=SIZE_TAG, weight=700, fill=key, spacing=1.2, fit=fit)
    if title is None:
        return fit, x0 + 14, y0 + 44
    g.text(x0 + 14, y0 + 44, title, size=SIZE_TITLE, weight=650, fit=fit)
    return fit, x0 + 14, y0 + 66


def arrow(g: Svg, d: str, key: str, width: float = 1.8, dash=None) -> None:
    g.path(d, stroke=key, width=width, marker=key, dash=dash)


def snowflake(g: Svg, x: float, y: float, r: float, key: str, width: float = 1.4) -> None:
    for a in (0, 60, 120):
        dx, dy = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
        g.path(f"M {x - dx:.1f} {y - dy:.1f} L {x + dx:.1f} {y + dy:.1f}", stroke=key, width=width, overlay=True)


def defs(g: Svg) -> None:
    p = g.p
    cx, cy, r = POCKET
    markers = []
    for key in ("s", "q", "t", "ink", "ink3"):
        markers.append(
            f'<marker id="arrow-{key}" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="9" markerHeight="9" '
            f'markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
            f'<path d="M1,1.2 L9,5 L1,8.8 L2.8,5 z" fill="{p[key]}"/></marker>'
        )
    g.add(
        "<defs>"
        + "".join(markers)
        + '<radialGradient id="metal" cx="38%" cy="34%" r="70%">'
          '<stop offset="0" stop-color="#fef3c7"/><stop offset="1" stop-color="#f59e0b"/></radialGradient>'
        + f'<radialGradient id="halo"><stop offset="0" stop-color="{p["t_glow"]}" stop-opacity="0.45"/>'
          f'<stop offset="1" stop-color="{p["t_glow"]}" stop-opacity="0"/></radialGradient>'
        + '<pattern id="frost" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
          f'<rect width="7" height="7" fill="{p["q_fill"]}"/><line x1="0" y1="0" x2="0" y2="7" '
          f'stroke="{p["q_frost"]}" stroke-width="3"/></pattern>'
        + f'<clipPath id="pocket-clip"><circle cx="{cx}" cy="{cy}" r="{r - 1}"/></clipPath>'
        + "</defs>"
    )


def stage_header(g: Svg) -> None:
    stages = [("A", "Input"), ("B", "Featurize"), ("C", "Encode"), ("D", "Pool"), ("E", "Fuse"), ("F", "Classify")]
    for i, (col, label) in enumerate(stages, start=1):
        x0, x1 = COL[col]
        mid = (x0 + x1) / 2
        width = 24 + len(label) * 9.2
        start = mid - width / 2
        g.circle(start + 9, 22, 9, fill="ink")
        g.text(start + 9, 26.2, str(i), size=11.5, weight=700, fill="on", anchor="middle", overlay=True)
        g.text(start + 24, 26.5, label.upper(), size=12, weight=700, fill="ink2", spacing=1.2,
               fit=(x0 - 12, 8, x1 + 12, 34))
    g.path(f"M {COL['A'][0]} 42 H {COL['F'][1]}", stroke="rule", width=1)


def catmull_rom(points: list[tuple[float, float]]) -> str:
    """Smooth path through the points (uniform Catmull-Rom converted to cubic Beziers)."""
    ext = [points[0]] + points + [points[-1]]
    d = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f" C {c1[0]:.1f} {c1[1]:.1f}, {c2[0]:.1f} {c2[1]:.1f}, {p2[0]:.1f} {p2[1]:.1f}"
    return d


def boundary(angle_deg: float) -> tuple[float, float]:
    cx, cy, r = POCKET
    a = math.radians(angle_deg)
    return cx + r * math.cos(a), cy + r * math.sin(a)


CHAIN_TAIL = (166, Q_MID)  # where the drawn chain ends outside the pocket; the sequence arrow starts here


def input_panel(g: Svg) -> None:
    cx, cy, r = POCKET
    px_per_a = r / 10.0
    x0, x1 = COL["A"]
    fit = (x0 - 2, 44, x1 + 2, LANE_T[1])
    g.text(cx, 78, "Holo structure", size=SIZE_TITLE, weight=650, anchor="middle", fit=fit)
    g.text(cx, 97, "+ metal position", size=SIZE_BODY, fill="ink2", anchor="middle", fit=fit)
    g.text(cx, 113, "(atom ID or x, y, z)", size=12, fill="txt3", anchor="middle", fit=fit)
    # Key to the drawing.
    ky = 140
    g.circle(x0 + 7, ky - 4.5, 6, fill="s", stroke="card", width=1)
    g.text(x0 + 19, ky, "first-shell residue", size=12, fill="ink2", fit=fit)
    g.circle(x0 + 7, ky + 12.5, 5, fill="card", stroke="s", width=2)
    g.text(x0 + 19, ky + 17, "second-shell residue", size=12, fill="ink2", fit=fit)
    g.circle(x0 + 7, ky + 29.5, 4.5, fill="s_node", stroke="s", width=1.2)
    g.text(x0 + 19, ky + 34, "other pocket residue", size=12, fill="ink2", fit=fit)
    g.circle(x0 + 7, ky + 46.5, 6, fill="url(#metal)")
    g.text(x0 + 19, ky + 51, "metal: position used,", size=12, fill="ink2", fit=fit)
    g.text(x0 + 19, ky + 66, "element never used", size=12, weight=650, fill="ink", fit=fit)
    g.circle(x0 + 4, ky + 79.5, 3.2, fill="#3b6fe0")
    g.circle(x0 + 11, ky + 79.5, 3.2, fill="#e5484d")
    g.text(x0 + 19, ky + 84, "N, O donor atoms", size=12, fill="ink2", fit=fit)
    g.text(cx, cy - r - 9, "any atom ≤ 10 Å", size=12.5, weight=600, fill="txt3", anchor="middle", fit=fit)

    g.circle(cx, cy, r, fill="card", stroke="ink3", width=1.1)
    g.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{g.c("card")}" stroke-width="1.6" '
          'stroke-dasharray="2.5 4"/>')
    k = r / 80
    # Schematic pocket (dx, dy, shell, one-letter code, donor element): a 3-His-1-Glu first shell, which
    # every metal class can bind, a second-shell Asp next to a His, and other pocket residues.
    residues = [
        (-34, -24, 1, "H", "N"), (33, -26, 1, "H", "N"), (30, 30, 1, "E", "O"), (-28, 32, 1, "H", "N"),
        (-62, -30, 2, "D", ""), (-6, -66, 0, "", ""), (48, -60, 0, "", ""), (68, 4, 0, "", ""),
        (56, 54, 0, "", ""), (4, 68, 0, "", ""), (-52, 58, 0, "", ""), (-70, 4, 0, "", ""),
    ]
    residues = [(dx * k, dy * k, shell, label, donor) for dx, dy, shell, label, donor in residues]
    pts = [(cx + dx, cy + dy) for dx, dy, *_ in residues]
    # Backbone: enters on the left, threads the pocket residues, leaves at the lower right and continues
    # outside the 10 A sphere (the rest of the chain, which the language model also sees).
    order = (11, 10, 3, 0, 4, 5, 6, 1, 7, 2, 9, 8)
    inside = [(cx - 104, cy + 10)] + [pts[i] for i in order]
    exit_pt = boundary(48)
    g.add('<g clip-path="url(#pocket-clip)">')
    g.path(catmull_rom(inside + [exit_pt, (exit_pt[0] + 14, exit_pt[1] + 26)]), stroke="back", width=4,
           opacity=0.38, overlay=True)
    g.add("</g>")
    g.add(f'<g><clipPath id="outside-clip"><path d="M 0 0 H {W} V {H} H 0 Z M {cx + r} {cy} '
          f'A {r} {r} 0 1 0 {cx - r} {cy} A {r} {r} 0 1 0 {cx + r} {cy} Z" clip-rule="evenodd"/></clipPath></g>')
    g.add('<g clip-path="url(#outside-clip)">')
    tx, ty = CHAIN_TAIL
    g.path(f"M {pts[8][0]:.1f} {pts[8][1]:.1f} L {exit_pt[0]:.1f} {exit_pt[1]:.1f} "
           f"C {exit_pt[0] - 2:.1f} {exit_pt[1] + 40:.1f}, {tx - 30:.1f} {ty:.1f}, {tx:.1f} {ty:.1f}",
           stroke="back", width=4, opacity=0.8, overlay=True)
    g.add("</g>")
    g.text(tx - 2, ty + 20, "rest of chain", size=11.5, fill="txt3", italic=True, anchor="end", fit=fit)
    # Graph edges (closest atoms <= 8 A). Residues binding the same metal always qualify, so every
    # first-shell pair is linked; the drawing links other pairs by proximity.
    g.add('<g clip-path="url(#pocket-clip)">')
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            both_first = residues[i][2] == 1 and residues[j][2] == 1
            if both_first or math.dist(pts[i], pts[j]) < 60 * k:
                g.path(f"M {pts[i][0]:.1f} {pts[i][1]:.1f} L {pts[j][0]:.1f} {pts[j][1]:.1f}",
                       stroke="s_soft", width=1.4, overlay=True)
    g.add("</g>")
    # Donor atoms (N blue, O red) of the first-shell residues, 2.1 A from the metal (to scale).
    donor_colour = {"N": "#3b6fe0", "O": "#e5484d"}
    for dx, dy, shell, label, donor in residues:
        if shell != 1:
            continue
        ang = math.atan2(dy, dx)
        dxp, dyp = cx + 2.1 * px_per_a * math.cos(ang), cy + 2.1 * px_per_a * math.sin(ang)
        nx, ny = cx + dx, cy + dy
        g.path(f"M {nx:.1f} {ny:.1f} L {dxp:.1f} {dyp:.1f}", stroke="s", width=2.6, overlay=True)
        g.circle(dxp, dyp, 3.2, fill=donor_colour[donor], stroke="card", width=0.8)
    for (_, _, shell, label, _), (px, py) in zip(residues, pts):
        if shell == 1:
            g.circle(px, py, 11, fill="s", stroke="card", width=1.5)
            g.text(px, py + 4.3, label, size=12, weight=700, fill="on", anchor="middle", overlay=True)
        elif shell == 2:
            g.circle(px, py, 10, fill="card", stroke="s", width=2.2)
            g.text(px, py + 4.3, label, size=12, weight=700, fill="s", anchor="middle", overlay=True)
        else:
            g.circle(px, py, 6.5, fill="s_node", stroke="s", width=1.2)
    g.circle(cx, cy, 9.5, fill="url(#metal)", stroke="card", width=1.3)
    g.text(cx, cy + 4.6, "?", size=13, weight=800, fill="#1f2328", anchor="middle", overlay=True)


def pool_box(g: Svg, key: str, per_residue: str, out: str, mid: float) -> None:
    """Pooling recipe; both branches share this layout so they can be compared line by line."""
    x0, x1 = COL["D"]
    height = 144
    y0, y1 = mid - height / 2, mid + height / 2
    shadowed(g, x0, y0, x1, y1, r=10, fill=f"{key}_fill", stroke=key)
    fit = (x0 + 5, y0, x1 - 5, y1)
    cxm = (x0 + x1) / 2
    g.text(cxm, y0 + 20, "Pool residues", size=13.5, weight=650, anchor="middle", fit=fit)
    g.text(cxm, y0 + 38, per_residue, size=13, fill="ink2", anchor="middle", fit=fit)
    # Glyph: one vector per pocket residue (… = any number) pooled into one vector.
    gy0, gy1 = y0 + 48, y0 + 70
    bx = cxm - 46
    for i in range(3):
        g.rect(bx + i * 9, gy0, bx + i * 9 + 6, gy1, r=1.5, fill=key)
    g.text(bx + 34, gy1 - 6, "…", size=13, fill="ink2", anchor="middle", overlay=True)
    g.rect(bx + 42, gy0, bx + 48, gy1, r=1.5, fill=key)
    arrow(g, f"M {bx + 55} {(gy0 + gy1) / 2} H {bx + 75}", key, 1.6)
    g.rect(bx + 82, gy0, bx + 90, gy1, r=1.5, fill=key)
    g.text(cxm, y0 + 90, "mean & attention", size=13, fill="ink2", anchor="middle", fit=fit)
    g.text(cxm, y0 + 108, "2×128 → 128", size=13, fill="ink2", anchor="middle", fit=fit)
    g.text(cxm, y1 - 12, out, size=15, weight=700, fill=key, anchor="middle", raw=True, fit=fit)


def structure_lane(g: Svg) -> None:
    fit, x, y = card(g, "B", LANE_S, "s", "STRUCTURE", "Pocket graph")
    sections = [("NODES", "protein residues",
                 ["any atom ≤ 10 Å from the metal", "scalars: residue type, chemistry,",
                  "shell flags, 3 distances to metal,", "ligand-orientation cosine",
                  "vectors: Cβ→side chain, metal→Cα"]),
                ("EDGES", "contacts",
                 ["closest atoms ≤ 8 Å apart", "contact and Cα–Cα distances,", "sequence separation, same chain,",
                  "contact vector"])]
    yy = y
    for n, (label, what, details) in enumerate(sections):
        if n:
            g.path(f"M {x} {yy - 12} H {COL['B'][1] - 14}", stroke="s_soft", width=1)
            yy += 5
        g.text(x, yy, f'<tspan font-weight="700" fill="{g.c("s")}" letter-spacing="1.1" font-size="11.5">'
                      f'{label}</tspan><tspan dx="7" font-weight="650">{escape(what)}</tspan>', size=13.5,
               raw=True, fit=fit)
        yy += 18
        for detail in details:
            g.text(x + 10, yy, detail, size=12.5, fill="ink2", fit=fit)
            yy += 16.5
        yy += 9

    fit, x, y = card(g, "C", LANE_S, "s", "STRUCTURE ENCODER", "GVP-GNN")
    g.text(x, y, "4 message-passing layers", fill="ink2", fit=fit)
    # Four plates = four layers. The front plate shows one layer acting on a residue i: each contact j
    # sends a message along its edge, and i is updated from the sum of its messages.
    px0, py0, pw, ph = x + 1, y + 12, 162, 62
    for k in range(4):
        off = 3 - k
        g.rect(px0 + 3 * off, py0 + 5 * off, px0 + 3 * off + pw, py0 + 5 * off + ph, r=7,
               fill="card", stroke="s", width=1.1)
    ix, iy = px0 + 79, py0 + 31
    nbrs = [(px0 + 20, py0 + 15), (px0 + 17, py0 + 46), (px0 + 52, py0 + 52), (px0 + 56, py0 + 11)]
    g.path(f"M {nbrs[0][0]} {nbrs[0][1]} L {nbrs[1][0]} {nbrs[1][1]} L {nbrs[2][0]} {nbrs[2][1]}",
           stroke="s_soft", width=1.3)
    for nx, ny in nbrs:
        ang = math.atan2(iy - ny, ix - nx)
        sx, sy = nx + 7.5 * math.cos(ang), ny + 7.5 * math.sin(ang)
        ex, ey = ix - 11.5 * math.cos(ang), iy - 11.5 * math.sin(ang)
        arrow(g, f"M {sx:.1f} {sy:.1f} L {ex:.1f} {ey:.1f}", "s", 1.5)
    for nx, ny in nbrs:
        g.circle(nx, ny, 5.5, fill="s_node", stroke="s", width=1.2)
    g.circle(ix, iy, 9, fill="s_node", stroke="s", width=2.4)
    g.text(ix, iy + 4.5, "i", size=12.5, weight=700, fill="s", anchor="middle", italic=True, overlay=True)
    pfit = (px0 + 2, py0 + 2, px0 + pw - 4, py0 + ph - 2)
    g.text(ix + 16, iy - 9, "sum of", size=12, fill="ink2", fit=pfit)
    g.text(ix + 16, iy + 5, "messages", size=12, fill="ink2", fit=pfit)
    g.text(ix + 16, iy + 19, "updates i", size=12, fill="ink2", fit=pfit)
    g.text(x, y + 116, "per residue: 128 scalars", fill="ink2", fit=fit)
    g.text(x, y + 133, "+ 16 vectors (3D)", fill="ink2", fit=fit)
    g.text(x, y + 152, "vectors rotate with protein;", size=12, fill="txt3", italic=True, fit=fit)
    g.text(x, y + 167, "scalars do not", size=12, fill="txt3", italic=True, fit=fit)
    g.text(x, y + 182, "(only scalars are pooled)", size=12, fill="txt3", italic=True, fit=fit)

    pool_box(g, "s", "128 scalars each", sub("h", "struct"), S_MID)

    ax, ay = boundary(-25)
    arrow(g, f"M {ax:.1f} {ay:.1f} C {ax + 18:.1f} {ay - 18:.1f}, {COL['B'][0] - 26} {S_MID}, "
             f"{COL['B'][0] - 2} {S_MID}", "s")
    arrow(g, f"M {COL['B'][1]} {S_MID} H {COL['C'][0] - 2}", "s")
    arrow(g, f"M {COL['C'][1]} {S_MID} H {COL['D'][0] - 2}", "s")


def sequence_lane(g: Svg) -> None:
    fit, x, y = card(g, "B", LANE_Q, "q", "SEQUENCE", "Chain sequences")
    g.text(x, y, "one sequence per chain,", fill="ink2", fit=fit)
    g.text(x, y + 18, "observed residues only", fill="ink2", fit=fit)
    cw = 11.2
    cells = ["…"] + list(EXAMPLE_SEQUENCE) + ["…"]
    for i, ch in enumerate(cells):
        cx = x + i * cw
        pocket = (i - 1) in EXAMPLE_POCKET
        if pocket:
            g.rect(cx - 1, y + 30, cx + cw - 1.2, y + 49, r=2.5, fill="q")
        g.text(cx + cw / 2 - 1, y + 44, ch, size=12.5, weight=650 if pocket else 400,
               fill="on" if pocket else "ink2", anchor="middle", fit=fit, overlay=pocket, mono=True)
    g.rect(x, y + 61, x + 10, y + 71, r=2, fill="q")
    g.text(x + 16, y + 70.5, "pocket residue", size=12, fill="txt3", fit=fit)

    fit, x, y = card(g, "C", LANE_Q, "q", "SEQUENCE ENCODER", "ESM-C 600M", fill="url(#frost)")
    bx, by = COL["C"][1] - 82, LANE_Q[0] - 10
    g.rect(bx - 2, by - 2, bx + 66, by + 21, r=11.5, fill="card")
    g.rect(bx, by, bx + 64, by + 19, r=9.5, fill="q", extra='data-avoid="1"')
    snowflake(g, bx + 11.5, by + 9.5, 5.4, "on")
    g.text(bx + 39.5, by + 13.5, "frozen", size=11.5, weight=700, fill="on", anchor="middle",
           fit=(bx + 19, by, bx + 63, by + 19), overlay=True)
    g.text(x, y, "protein language model", fill="ink2", fit=fit)
    g.text(x, y + 18, "per residue: 1152 values,", fill="ink2", fit=fit)
    g.text(x, y + 36, "with whole-chain context", fill="ink2", fit=fit)
    cells = 16
    cw = 10.6
    for i in range(cells):
        cx = x + i * cw
        if i in (0, cells - 1):
            g.text(cx + cw / 2 - 1, y + 58, "…", size=12.5, fill="ink2", anchor="middle", fit=fit)
            continue
        g.rect(cx, y + 48, cx + 9, y + 59, r=1.5, fill="q", opacity=0.95 if (i - 1) in EXAMPLE_POCKET else 0.25)

    pool_box(g, "q", "1152 → 128 each", sub("h", "seq"), Q_MID)

    # The pooled embeddings come from the same residues as the graph nodes.
    gap_y = (LANE_S[1] + LANE_Q[0]) / 2
    bx_mid = (COL["B"][0] + COL["B"][1]) / 2
    dx_mid = (COL["D"][0] + COL["D"][1]) / 2
    lx = (COL["C"][0] + COL["C"][1]) / 2
    half = 72
    g.path(f"M {bx_mid} {LANE_S[1] + 3} V {gap_y} H {lx - half - 6}", stroke="ink3", width=1.4, dash="1.5 4")
    arrow(g, f"M {lx + half + 6} {gap_y} H {dx_mid} V {Q_MID - 72 - 3}", "ink3", 1.4, dash="1.5 4")
    g.text(lx, gap_y + 4.5, "same pocket residues", size=12.5, fill="txt3", anchor="middle", italic=True,
           fit=(lx - half, gap_y - 12, lx + half, gap_y + 12))

    # Sequence arrow starts on the chain outside the pocket: each chain is embedded in full.
    tx, ty = CHAIN_TAIL
    arrow(g, f"M {tx + 4} {ty} H {COL['B'][0] - 2}", "q")
    arrow(g, f"M {COL['B'][1]} {Q_MID} H {COL['C'][0] - 2}", "q")
    arrow(g, f"M {COL['C'][1]} {Q_MID} H {COL['D'][0] - 2}", "q")


def site_lane(g: Svg) -> None:
    fit, x, y = card(g, "B", LANE_T, "t", "SITE", "Site summary")
    g.text(COL["B"][1] - 14, LANE_T[0] + 44, "4 values", size=12.5, fill="ink2", anchor="end", fit=fit)
    g.text(x, y, "multinuclear flag, metal count,", size=12.5, fill="ink2", fit=fit)
    g.text(x, y + 17, "min and mean metal–metal distance", size=12.5, fill="ink2", fit=fit)
    g.text(x, y + 36, "constant: one metal per example", size=12, fill="txt3", italic=True, fit=fit)

    ex0, ex1 = COL["C"]
    efit, ex, ey = card(g, "C", LANE_T, "t", "SITE ENCODER", "Linear layer")
    g.text(ex, ey - 1, "4 → 32", fill="ink2", fit=efit)
    g.text(ex1 - 14, ey - 1, sub("h", "site"), size=15, weight=700, fill="t", anchor="end", raw=True, fit=efit)
    g.text(ex, ey + 18, "constant input, so it", size=12, fill="txt3", italic=True, fit=efit)
    g.text(ex, ey + 34, "acts as a learned bias", size=12, fill="txt3", italic=True, fit=efit)

    ax, ay = boundary(118)
    x0 = COL["B"][0]
    arrow(g, f"M {ax:.1f} {ay:.1f} C {ax - 6:.1f} {ay + 120:.1f}, {x0 - 70} {T_MID}, {x0 - 2} {T_MID}", "t")
    arrow(g, f"M {COL['B'][1]} {T_MID} H {ex0 - 2}", "t")


def fusion(g: Svg) -> None:
    dx1 = COL["D"][1]
    bar_x0, bar_x1, top, unit = BAR
    s_end = top + 128 * unit
    q_end = s_end + 128 * unit
    t_end = q_end + 32 * unit
    bar_mid = (bar_x0 + bar_x1) / 2
    arrow(g, f"M {dx1} {S_MID} H {bar_x0 - 2}", "s")
    # Gate = sigmoid(W [h_struct ; h_seq]) scales h_seq element-wise.
    gate_y = (S_MID + Q_MID) / 2
    g.circle(TAP_X, S_MID, 3, fill="s")
    g.circle(TAP_X, Q_MID, 3, fill="q")
    g.path(f"M {TAP_X} {S_MID} V {gate_y - 18} Q {TAP_X} {gate_y - 8} {GATE_X - 13} {gate_y - 6}", stroke="s",
           width=1.4, marker="s")
    g.path(f"M {TAP_X} {Q_MID} V {gate_y + 18} Q {TAP_X} {gate_y + 8} {GATE_X - 13} {gate_y + 6}", stroke="q",
           width=1.4, marker="q")
    g.circle(GATE_X, gate_y, 13, fill="card", stroke="ink", width=1.5)
    g.text(GATE_X, gate_y + 5, "σ", size=15, weight=700, anchor="middle", overlay=True)
    g.text(GATE_X, gate_y - 20, "gate", size=12.5, weight=650, fill="ink2", anchor="middle",
           fit=(TAP_X - 4, gate_y - 36, bar_x0 - 2, gate_y - 13))
    arrow(g, f"M {GATE_X} {gate_y + 13} V {Q_MID - 12}", "ink", 1.6)
    g.path(f"M {dx1} {Q_MID} H {GATE_X - 10}", stroke="q", width=1.8)
    # Element-wise product (Hadamard), drawn as a dot in a circle.
    g.circle(GATE_X, Q_MID, 9.5, fill="card", stroke="ink", width=1.5)
    g.circle(GATE_X, Q_MID, 2.6, fill="ink")
    arrow(g, f"M {GATE_X + 9.5} {Q_MID} H {bar_x0 - 2}", "q")
    t_mid = (q_end + t_end) / 2
    elbow_x = 866
    arrow(g, f"M {COL['C'][1]} {T_MID} H {elbow_x - 8} Q {elbow_x} {T_MID} {elbow_x} {T_MID - 8} V {t_mid + 8} "
             f"Q {elbow_x} {t_mid} {elbow_x + 8} {t_mid} H {bar_x0 - 2}", "t")
    # Concatenated vector; segment heights are proportional to 128 : 128 : 32 values.
    g.rect(bar_x0, top, bar_x1, s_end, r=3, fill="s")
    g.rect(bar_x0, s_end - 3, bar_x1, q_end, r=0, fill="q")
    g.rect(bar_x0, q_end - 3, bar_x1, t_end, r=3, fill="t")
    g.path(f"M {bar_x0} {s_end} H {bar_x1} M {bar_x0} {q_end} H {bar_x1}", stroke="card", width=2, cap="butt",
           overlay=True)
    for y0, y1, label in ((top, s_end, "128"), (s_end, q_end, "128"), (q_end, t_end, "32")):
        ym = (y0 + y1) / 2
        g.add(f'<text x="{bar_mid}" y="{ym}" font-size="12" font-weight="700" fill="{g.c("on")}" '
              f'text-anchor="middle" dominant-baseline="central" class="mono" '
              f'transform="rotate(-90 {bar_mid} {ym})">{label}</text>')
    g.text(bar_mid, top - 22, "concat", size=12.5, weight=650, fill="ink2", anchor="middle",
           fit=(bar_mid - 32, top - 38, bar_mid + 32, top - 16))
    g.text(bar_mid, top - 7, "288", size=12.5, weight=700, fill="ink", anchor="middle",
           fit=(bar_mid - 32, top - 20, bar_mid + 32, top))
    # A brace over the whole bar: the classifier reads all 288 values.
    bx = bar_x1 + 5
    mid = (top + t_end) / 2
    g.path(f"M {bx} {top} Q {bx + 6} {top} {bx + 6} {top + 8} V {mid - 8} Q {bx + 6} {mid} {bx + 12} {mid} "
           f"Q {bx + 6} {mid} {bx + 6} {mid + 8} V {t_end - 8} Q {bx + 6} {t_end} {bx} {t_end}", stroke="ink",
           width=1.4)


def classifier(g: Svg) -> None:
    x0, x1 = COL["F"]
    bar_x1, top, unit = BAR[1], BAR[2], BAR[3]
    mid = top + (288 * unit) / 2
    y0, y1 = mid - 100, mid + 110
    arrow(g, f"M {bar_x1 + 18} {mid} H {x0 - 2}", "ink")
    shadowed(g, x0, y0, x1, y1, fill="f_fill", stroke="ink3")
    fit = (x0 + 8, y0, x1 - 8, y1)
    g.text(x0 + 14, y0 + 22, "CLASSIFIER", size=SIZE_TAG, weight=700, fill="ink2", spacing=1.2, fit=fit)
    g.text(x0 + 14, y0 + 44, "MLP head", size=SIZE_TITLE, weight=650, fit=fit)
    g.text(x0 + 14, y0 + 66, "288 → 128 → 5", size=SIZE_BODY, fill="ink2", fit=fit)
    g.text(x0 + 14, y0 + 84, "softmax over 5 classes", size=12.5, fill="txt3", fit=fit)
    # Element tiles in the model's class order.
    widths = [28, 28, 28, 28, 40]
    tx = x0 + 14
    ty0, ty1 = y0 + 96, y0 + 126
    centres = []
    for (symbol, number), w in zip(CLASSES, widths):
        g.rect(tx, ty0, tx + w, ty1, r=4, fill="tile", stroke="ink", width=1.1)
        g.text(tx + w / 2, ty1 - 10, symbol, size=14 if len(symbol) <= 2 else 12, weight=700, anchor="middle",
               fit=(tx + 1, ty0, tx + w - 1, ty1))
        centres.append(tx + w / 2)
        tx += w + 3.5
    fx, cx = centres[3], centres[4]
    by = ty1 + 10
    g.path(f"M {fx} {ty1 + 4} V {by - 4} Q {fx} {by} {fx + 4} {by} H {cx - 4} Q {cx} {by} {cx} {by - 4} V {ty1 + 4}",
           stroke="ink3", width=1.3)
    g.path(f"M {(fx + cx) / 2} {by} V {by + 6}", stroke="ink3", width=1.3)
    g.text((fx + cx) / 2, by + 20, "Class VIII", size=12, weight=650, fill="ink2", anchor="middle", fit=fit)
    g.text(x0 + 14, by + 46, "4-class scores:", size=12.5, weight=650, fill="ink2", fit=fit)
    g.text(x0 + 14, by + 64, "Mn, Cu, Zn, Class VIII", size=12.5, fill="ink2", fit=fit)


def footer(g: Svg, *, training: bool = True) -> None:
    y = LANE_T[1] + 48
    g.path(f"M {COL['A'][0]} {y - 22} H {COL['F'][1]}", stroke="rule", width=1)
    x = COL["A"][0]
    fit = (COL["A"][0] - 3, y - 18, COL["F"][1], H)
    for key, label, x in (("s", "structure", 24), ("q", "sequence", 132), ("t", "site (constant)", 238)):
        g.rect(x, y - 9.5, x + 12, y + 2.5, r=3, fill=key)
        g.text(x + 18, y + 2, label, size=12.5, fill="ink2", fit=fit)
    x = 380
    g.rect(x, y - 10, x + 22, y + 3, r=6.5, fill="q")
    snowflake(g, x + 11, y - 3.5, 4.4, "on", 1.3)
    g.text(x + 28, y + 2, "frozen; all other modules are trained together", size=12.5, fill="ink2", fit=fit)
    g.text(COL["A"][0], y + 24,
           "σ gate: a 0–1 weight for each of the 128 h_seq values, computed from h_struct and h_seq  ·  "
           "⊙: element-wise product  ·  attention: weighted average, a learned score per residue", size=12,
           fill="txt3", fit=fit)
    g.text(COL["A"][0], y + 42,
           "First shell: side-chain donor atom ≤ 2.7 Å from the metal  ·  second shell: side-chain group ≤ 4.5 Å "
           "from a first-shell group  ·  not used: waters, small molecules, other metals", size=12, fill="txt3", fit=fit)
    g.text(COL["A"][0], y + 60,
           f"Shown: one network of public release {RELEASE}, which averages the 4-class scores of five such "
           "networks, each trained with a different CV fold held out.", size=12, fill="txt3", fit=fit)
    g.text(COL["A"][0], y + 78, "Drawings are schematic; a real pocket has about 38 residues (training median).",
           size=12, fill="txt3", fit=fit)
    if training:
        g.text(COL["A"][0], y + 78,
               "Training: class-weighted cross-entropy · AdamW, weight decay 1e-4 · learning rate 1e-4 for the "
               "GVP layers and node/edge embeddings, 3e-5 for the rest · cosine, 50 epochs, batch 16",
               size=12, fill="txt3", fit=fit)


def build(theme: str) -> str:
    g = Svg(THEMES[theme])
    defs(g)
    stage_header(g)
    input_panel(g)
    structure_lane(g)
    sequence_lane(g)
    site_lane(g)
    fusion(g)
    classifier(g)
    footer(g, training=False)
    title = "DeepMzyme architecture"
    desc = ("A holo protein structure and a supplied metal position (the metal's element is never used) feed "
            "three branches. Structure: a graph of the protein residues with any atom within 10 Å of the metal, "
            "linked when their closest atoms are within 8 Å, encoded by a four-layer GVP-GNN and pooled over the "
            "pocket residues. Sequence: each chain embedded in full by frozen ESM-C 600M and pooled over the same "
            "pocket residues. Site: a four-value summary that is constant in this release. A sigmoid gate computed "
            "from the two pooled vectors scales the sequence vector; the three vectors are concatenated (288 "
            "values) and an MLP gives softmax scores for five classes: Mn, Cu, Zn, Fe and Co/Ni. The release sums "
            "Fe and Co/Ni into Class VIII and averages the 4-class scores of five copies, each trained with one "
            "CV fold held out.")
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'role="img" aria-labelledby="title desc" font-family="{escape(SANS)}">'
            f'<title id="title">{title}</title><desc id="desc">{escape(desc)}</desc>'
            f'<style>.mono{{font-family:{MONO};}} text{{font-kerning:normal;}}</style>')
    return head + "\n".join(g.out) + "</svg>\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "docs" / "assets")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = args.out_dir / f"deepmzyme_architecture_{theme}.svg"
        path.write_text(build(theme), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
