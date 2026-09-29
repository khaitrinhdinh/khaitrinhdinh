"""Turn an image into an animated dot-matrix SVG.

The SVG animates with pure CSS @keyframes (no JavaScript, no external assets),
so GitHub renders it inside a README via <img>.

Effects:
    reveal  dots pop in one by one (order set by --order), hold, then fade out
    wave    the whole picture ripples like a flag in the wind
    fly     the picture glides across a wide banner, bobbing as it goes

Usage:
    python tools/dotart.py input.png -o assets/dots.svg --cols 48 --order wave
    python tools/dotart.py flag.png -o assets/flag.svg --effect wave --grid "" --pole
    python tools/dotart.py bird.png -o assets/bird.svg --effect fly --tint "#5a3510,#f2c46d"
"""
import argparse
import math
import random
from collections import Counter

from PIL import Image, ImageDraw

DELAY_STEPS = 48  # delays are bucketed into this many CSS classes to keep the file small


def load_cells(path, cols, n_colors, alpha_cut, key_tol, tint=None):
    """Downsample the image to a cols x rows grid; return (rows, {(x, y): (r, g, b)})."""
    img = Image.open(path).convert("RGBA")
    rows = max(1, round(cols * img.height / img.width))
    small = img.resize((cols, rows), Image.BOX)  # BOX = plain average of each cell

    px = small.load()
    has_alpha = any(px[x, y][3] < 255 for y in range(rows) for x in range(cols))
    key = None
    if not has_alpha:
        # Opaque image: assume the most common corner colour is the background.
        corners = [px[0, 0], px[cols - 1, 0], px[0, rows - 1], px[cols - 1, rows - 1]]
        key = Counter(c[:3] for c in corners).most_common(1)[0][0]

    cells = {}
    for y in range(rows):
        for x in range(cols):
            r, g, b, a = px[x, y]
            if a < alpha_cut:
                continue
            if key and math.dist((r, g, b), key) < key_tol:
                continue
            cells[(x, y)] = (r, g, b)

    if tint:
        cells = apply_tint(cells, *tint)
    if n_colors and cells:
        cells = quantize(cells, n_colors)
    return rows, cells


def parse_hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def apply_tint(cells, dark, light):
    """Recolour by brightness: black maps to `dark`, white to `light` (e.g. a bronze ramp)."""
    out = {}
    for k, (r, g, b) in cells.items():
        t = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        out[k] = tuple(round(d + (l - d) * t) for d, l in zip(dark, light))
    return out


def quantize(cells, n_colors):
    """Reduce to a small palette so the picture reads as flat cartoon colours."""
    keys = list(cells)
    strip = Image.new("RGB", (len(keys), 1))
    strip.putdata([cells[k] for k in keys])
    q = strip.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
    return dict(zip(keys, q.getdata()))


def value_noise(cols, rows, scale, rng):
    """Smooth random field in [0, 1] (bilinear-interpolated coarse grid)."""
    gw, gh = cols // scale + 2, rows // scale + 2
    grid = [[rng.random() for _ in range(gw)] for _ in range(gh)]

    def at(x, y):
        fx, fy = x / scale, y / scale
        x0, y0 = int(fx), int(fy)
        tx, ty = fx - x0, fy - y0
        tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)  # smoothstep
        top = grid[y0][x0] * (1 - tx) + grid[y0][x0 + 1] * tx
        bot = grid[y0 + 1][x0] * (1 - tx) + grid[y0 + 1][x0 + 1] * tx
        return top * (1 - ty) + bot * ty

    return at


def reveal_order(cells, cols, rows, mode, seed):
    """Map each cell to a value in [0, 1]: 0 lights up first, 1 last."""
    rng = random.Random(seed)
    cx, cy = (cols - 1) / 2, (rows - 1) / 2
    max_d = math.hypot(cx, cy) or 1
    noise = value_noise(cols, rows, 6, rng) if mode == "ink" else None

    def raw(x, y, rgb):
        if mode == "wave":
            return math.hypot(x - cx, y - cy) / max_d
        if mode == "diagonal":
            return (x + y) / max(cols + rows - 2, 1)
        if mode == "scan":
            return (y * cols + x) / max(cols * rows - 1, 1)
        if mode == "random":
            return rng.random()
        if mode == "ink":
            return noise(x, y)
        if mode == "bright":
            r, g, b = rgb
            return 1 - (0.299 * r + 0.587 * g + 0.114 * b) / 255
        raise ValueError(mode)

    vals = {k: raw(*k, rgb) for k, rgb in cells.items()}
    lo, hi = min(vals.values()), max(vals.values())
    span = (hi - lo) or 1
    return {k: (v - lo) / span for k, v in vals.items()}


def mark(x, y, cls, cell, shape, dy=0.0):
    """One dot at grid cell (x, y), shifted down by dy px."""
    if shape == "square":
        side = cell * 0.84
        return (f'<rect class="{cls}" x="{x * cell + (cell - side) / 2:.1f}" '
                f'y="{y * cell + (cell - side) / 2 + dy:.1f}" width="{side:.1f}" '
                f'height="{side:.1f}" rx="{side * 0.2:.1f}"/>')
    return (f'<circle class="{cls}" cx="{x * cell + cell / 2:.1f}" '
            f'cy="{y * cell + cell / 2 + dy:.1f}" r="{cell * 0.42:.1f}"/>')


def palette_css(cells):
    palette = {c: i for i, c in enumerate(sorted(set(cells.values())))}
    return palette, [".c%d{fill:#%02x%02x%02x}" % (i, *c) for c, i in palette.items()]


def build_wave_svg(cells, cols, rows, cell, shape, grid_color, amp, waves, period, pole):
    """Flag-in-the-wind: each column bobs on a sine wave, phase-shifted along x.

    Amplitude grows away from the pole (left edge), and each column is shaded
    darker/lighter in step with its slope so the folds read as 3D.
    """
    amp_px = amp * cell
    pad = math.ceil(amp_px) + cell  # room above/below for the ripple
    left = 2 if pole else 0  # columns reserved for the pole
    pole_len = rows // 2 if pole else 0  # pole continues below the flag
    w, h = (cols + left) * cell, (rows + pole_len) * cell + 2 * pad

    css = [f".col{{animation:bob {period:.2f}s ease-in-out infinite alternate,"
           f"shade {period:.2f}s ease-in-out infinite alternate}}",
           "@keyframes shade{from{opacity:1}to{opacity:.62}}"]
    body = []
    palette, pcss = palette_css(cells)
    css += pcss

    for x in range(cols):
        # Free end flaps harder than the end tied to the pole.
        a = amp_px * (0.15 + 0.85 * x / max(cols - 1, 1))
        # Negative delay = start mid-cycle, so columns are out of phase from frame 0.
        phase = -(x / cols) * waves * 2 * period
        css.append(f"@keyframes b{x}{{from{{transform:translateY({-a:.1f}px)}}"
                   f"to{{transform:translateY({a:.1f}px)}}}}")
        css.append(f".x{x}{{animation-name:b{x},shade;"
                   f"animation-delay:{phase:.2f}s,{phase - period / 2:.2f}s}}")
        dots = [mark(x + left, y, f"c{palette[cells[(x, y)]]}", cell, shape, pad)
                for y in range(rows) if (x, y) in cells]
        if dots:
            body.append(f'<g class="col x{x}">' + "".join(dots) + "</g>")

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
           f"<style>{''.join(css)}</style>"]
    if grid_color:
        out.append(f'<g fill="{grid_color}" fill-opacity=".15">')
        out += [mark(x + left, y, "g", cell, shape, pad) for y in range(rows) for x in range(cols)]
        out.append("</g>")
    if pole:
        # Static pole from just above the flag down past its bottom edge.
        out.append('<g fill="#c9d1d9">')
        pole_rows = range(-1, rows + pole_len)
        out += [mark(0, y, "", cell, shape, pad) for y in pole_rows]
        out.append("</g>")
    out += body
    out.append("</svg>")
    return "\n".join(out)


def build_fly_svg(cells, cols, rows, cell, shape, grid_color, width, duration, bob):
    """The picture flies left-to-right across a `width`-px banner and loops.

    An outer group slides horizontally; an inner group bobs, tilts and squashes
    slightly so it reads as flapping flight rather than a sliding sticker.
    """
    bird_w = cols * cell
    bob_px = bob * cell
    pad = math.ceil(bob_px) + cell * 2
    h = rows * cell + 2 * pad

    palette, pcss = palette_css(cells)
    css = [
        f".fly{{animation:fly {duration:.2f}s linear infinite}}",
        f"@keyframes fly{{from{{transform:translateX({-bird_w}px)}}to{{transform:translateX({width}px)}}}}",
        ".bob{transform-box:fill-box;transform-origin:center;"
        "animation:bob .9s ease-in-out infinite alternate}",
        f"@keyframes bob{{from{{transform:translateY({-bob_px:.1f}px) rotate(-2deg) scaleY(1)}}"
        f"to{{transform:translateY({bob_px:.1f}px) rotate(2deg) scaleY(.9)}}}}",
    ] + pcss

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {h}" width="{width}" height="{h}">',
           f"<style>{''.join(css)}</style>"]
    if grid_color:
        # One <pattern> instead of thousands of background dots keeps the file small.
        r = cell * 0.42
        out.append(f'<defs><pattern id="grid" width="{cell}" height="{cell}" patternUnits="userSpaceOnUse">'
                   f'<circle cx="{cell / 2}" cy="{cell / 2}" r="{r:.1f}" fill="{grid_color}" fill-opacity=".12"/>'
                   f'</pattern></defs><rect width="{width}" height="{h}" fill="url(#grid)"/>')
    out.append('<g class="fly"><g class="bob">')
    out += [mark(x, y, f"c{palette[rgb]}", cell, shape, pad) for (x, y), rgb in cells.items()]
    out.append("</g></g></svg>")
    return "\n".join(out)


def build_svg(cells, order, cols, rows, cell, shape, grid_color, spread, pop, hold, gap):
    w, h = cols * cell, rows * cell
    period = 2 * pop + 2 * spread + hold + gap

    def pct(t):
        return f"{100 * t / period:.2f}%"

    shown_until = pop + spread + hold
    css = [
        ".d{opacity:0;transform-box:fill-box;transform-origin:center;"
        f"animation:pop {period:.2f}s ease-in-out infinite both}}",
        "@keyframes pop{"
        "0%{opacity:0;transform:scale(.2)}"
        f"{pct(pop * 0.6)}{{opacity:1;transform:scale(1.3)}}"
        f"{pct(pop)}{{opacity:1;transform:scale(1)}}"
        f"{pct(shown_until)}{{opacity:1;transform:scale(1)}}"
        f"{pct(shown_until + pop)}{{opacity:0;transform:scale(.2)}}"
        "100%{opacity:0;transform:scale(.2)}}",
    ]
    for i in range(DELAY_STEPS):
        css.append(f".t{i}{{animation-delay:{spread * i / (DELAY_STEPS - 1):.3f}s}}")

    palette, pcss = palette_css(cells)
    css += pcss

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f"<style>{''.join(css)}</style>",
    ]
    if grid_color:
        out.append(f'<g fill="{grid_color}" fill-opacity=".15">')
        out += [mark(x, y, "g", cell, shape) for y in range(rows) for x in range(cols)]
        out.append("</g>")
    for (x, y), rgb in sorted(cells.items(), key=lambda kv: order[kv[0]]):
        step = round(order[(x, y)] * (DELAY_STEPS - 1))
        out.append(mark(x, y, f"d c{palette[rgb]} t{step}", cell, shape))
    out.append("</svg>")
    return "\n".join(out)


def render_preview(cells, cols, rows, cell, shape, path):
    """Static PNG of the fully revealed frame, for a quick look without a browser."""
    scale = 2
    img = Image.new("RGB", (cols * cell * scale, rows * cell * scale), (13, 17, 23))
    draw = ImageDraw.Draw(img)
    s = cell * scale
    pad = s * 0.08
    for y in range(rows):
        for x in range(cols):
            box = (x * s + pad, y * s + pad, (x + 1) * s - pad, (y + 1) * s - pad)
            fill = cells.get((x, y), (33, 38, 45))
            if shape == "square":
                draw.rounded_rectangle(box, radius=s * 0.17, fill=fill)
            else:
                draw.ellipse(box, fill=fill)
    img.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("-o", "--out", default="dots.svg")
    p.add_argument("--cols", type=int, default=48, help="dots per row")
    p.add_argument("--cell", type=int, default=12, help="px per dot cell")
    p.add_argument("--shape", choices=["circle", "square"], default="circle")
    p.add_argument("--effect", choices=["reveal", "wave", "fly"], default="reveal")
    p.add_argument("--order", choices=["wave", "diagonal", "scan", "random", "ink", "bright"], default="wave")
    p.add_argument("--colors", type=int, default=12, help="palette size (0 = keep original colours)")
    p.add_argument("--grid", default="#8b949e", help="colour of the empty dot grid ('' to hide)")
    p.add_argument("--spread", type=float, default=2.5, help="seconds from first to last dot lighting up")
    p.add_argument("--pop", type=float, default=0.5, help="seconds for one dot to pop in/out")
    p.add_argument("--hold", type=float, default=3.0, help="seconds the full picture stays visible")
    p.add_argument("--gap", type=float, default=0.8, help="blank pause before the loop restarts")
    p.add_argument("--amp", type=float, default=1.2, help="wave: max ripple height, in dots")
    p.add_argument("--waves", type=float, default=1.2, help="wave: ripples across the flag")
    p.add_argument("--period", type=float, default=1.1, help="wave: seconds per half ripple")
    p.add_argument("--pole", action="store_true", help="wave: draw a flag pole on the left")
    p.add_argument("--width", type=int, default=900, help="fly: banner width in px")
    p.add_argument("--duration", type=float, default=12, help="fly: seconds to cross the banner")
    p.add_argument("--bob", type=float, default=1.0, help="fly: bob height, in dots")
    p.add_argument("--tint", help='recolour by brightness, "dark_hex,light_hex" (e.g. bronze)')
    p.add_argument("--alpha-cut", type=int, default=128)
    p.add_argument("--key-tol", type=float, default=40, help="distance to background colour for opaque images (0 = keep background)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--preview", help="also write a static PNG of the final frame")
    a = p.parse_args()

    tint = tuple(parse_hex(c) for c in a.tint.split(",")) if a.tint else None
    rows, cells = load_cells(a.image, a.cols, a.colors, a.alpha_cut, a.key_tol, tint)
    if not cells:
        raise SystemExit("No foreground pixels found - try a lower --alpha-cut or --key-tol.")
    if a.effect == "fly":
        svg = build_fly_svg(cells, a.cols, rows, a.cell, a.shape, a.grid,
                            a.width, a.duration, a.bob)
    elif a.effect == "wave":
        svg = build_wave_svg(cells, a.cols, rows, a.cell, a.shape, a.grid,
                             a.amp, a.waves, a.period, a.pole)
    else:
        order = reveal_order(cells, a.cols, rows, a.order, a.seed)
        svg = build_svg(cells, order, a.cols, rows, a.cell, a.shape, a.grid,
                        a.spread, a.pop, a.hold, a.gap)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(svg)
    if a.preview:
        render_preview(cells, a.cols, rows, a.cell, a.shape, a.preview)
    print(f"{a.out}: {a.cols}x{rows} grid, {len(cells)} dots, {len(svg) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
