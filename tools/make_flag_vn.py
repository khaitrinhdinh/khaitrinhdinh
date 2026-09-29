"""Draw the Vietnamese flag (3:2, red field, centred yellow five-pointed star) as a PNG."""
import math
import sys

from PIL import Image, ImageDraw

W, H = 900, 600
RED, YELLOW = (218, 37, 29), (255, 221, 0)

img = Image.new("RGB", (W, H), RED)
cx, cy = W / 2, H / 2
outer = H * 0.3            # circumscribed radius of the star
inner = outer * 0.382      # regular pentagram inner/outer ratio
pts = []
for i in range(10):
    r = outer if i % 2 == 0 else inner
    ang = -math.pi / 2 + i * math.pi / 5  # first point straight up
    pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
ImageDraw.Draw(img).polygon(pts, fill=YELLOW)
img.save(sys.argv[1] if len(sys.argv) > 1 else "flag_vn.png")
