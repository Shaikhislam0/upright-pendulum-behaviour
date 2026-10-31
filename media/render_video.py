"""Render one complete archived trajectory; no controller inference or training.

Run from this directory after installing requirements-video.txt. The source CSV
has 20 ms state samples. Recorded hold completion comes from the 2 ms monitor;
the renderer never re-estimates the success event from its coarser samples.
"""

import csv
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

BASE = Path(__file__).resolve().parent
W, H, FPS = 1280, 720, 50
BG, INK, MUTED = "#faf9f5", "#202b32", "#5d686d"
TEAL, ORANGE, GRID = "#167367", "#ad5d21", "#d8dbd8"


def font(size, bold=False):
    # Matplotlib supplies the same portable fonts on Windows, Linux and macOS.
    import matplotlib
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf" / name), size)


def render():
    info = json.loads((BASE / "selection.json").read_text())
    with (BASE / "successful_trial.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    times = [float(r["time_s"]) for r in rows]
    assert len(rows) == 1001 and times[0] == 0 and times[-1] == 20
    xs = [float(r["x_m"]) for r in rows]
    angles = [math.degrees(math.atan2(math.sin(float(r["theta_rad_unwrapped"])),
                                    math.cos(float(r["theta_rad_unwrapped"])))) for r in rows]
    forces = [float(r["force_N"]) if r["force_N"] else 0 for r in rows]
    ts = info["success_time_s"]
    fs = {n: font(n) for n in (13, 14, 15, 16, 18, 21, 24, 30, 40)}
    fbold = font(25, True)
    # Fixed world scale and fixed camera: horizontal travel remains visible.
    extent = max(2.0, max(abs(x) for x in xs) + 1.12)
    scale = min(157., 388 / extent)
    center, py = 450, 318

    def px(x):
        return center + x * scale

    static = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(static)
    d.text((38, 22), "Cart-pendulum | swing-up and stabilization", font=fbold, fill=INK)
    d.text((39, 62), f"{info['reward'].capitalize()} reward   /   seed {info['training_seed']}   /   generation 1,000", font=fs[16], fill=MUTED)
    d.line((38, 98, 1242, 98), fill=GRID, width=2)
    d.text((40, 115), "FULL ARCHIVED EVALUATION  /  REAL-TIME PLAYBACK", font=fs[13], fill=MUTED)
    # Goal region has actual +/- 0.1 m width; ghost rod is the upright target.
    d.rectangle((px(-.1), py - 164, px(.1), py + 67), fill="#eaf0e6")
    d.line((px(0), py - scale, px(0), py), fill="#b1beb7", width=2)
    rail_y = py + 61
    d.line((46, rail_y, 856, rail_y), fill="#727c80", width=2)
    for x in range(math.ceil(-extent), math.floor(extent) + 1):
        if 48 < px(x) < 854:
            d.line((px(x), rail_y, px(x), rail_y + 7), fill=MUTED, width=1)
            if x != 0:
                d.text((px(x) - 12, rail_y + 12), f"{x} m", font=fs[13], fill=MUTED)
    d.text((42, 490), "Fixed camera; rod length 1 m. Shaded cart target: |x| <= 0.10 m.", font=fs[14], fill=MUTED)
    d.line((891, 115, 891, 495), fill=GRID, width=1)
    # Plot scales are held fixed over the full clip.
    charts = [("Cart position (m)", xs, max(.2, max(abs(x) for x in xs) * 1.1), TEAL),
              ("Upright angle error (deg)", angles, 180, INK),
              ("Applied force (N)", forces, 10, ORANGE)]
    panel_geometries = []
    for j, (label, values, vmax, color) in enumerate(charts):
        left, right = 44 + j * 414, 418 + j * 414
        top, bottom = 551, 651
        d.text((left, 516), label, font=fs[14], fill=INK)
        d.rectangle((left, top, right, bottom), outline=GRID)
        mid = (top + bottom) / 2
        d.line((left, mid, right, mid), fill=GRID)
        points = [(left + t / 20 * (right-left), mid - v / vmax * 46) for t, v in zip(times, values)]
        # Break the wrapped-angle trace at its coordinate discontinuity.
        for k in range(1, len(points)):
            if j == 1 and abs(values[k] - values[k-1]) > 180:
                continue
            d.line((points[k-1], points[k]), fill="#c7ceca", width=1)
        d.text((left, bottom+5), "0 s", font=fs[13], fill=MUTED)
        d.text((right-30, bottom+5), "20 s", font=fs[13], fill=MUTED)
        panel_geometries.append((left, right, top, bottom, points, color))
    d.text((41, 694), "Illustrative successful trial; not a measure of cohort success. State samples: 20 ms; event monitor: 2 ms.", font=fs[13], fill=MUTED)

    output = BASE / "successful_simulation.mp4"
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
               "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        for i, r in enumerate(rows):
            image = static.copy()
            draw = ImageDraw.Draw(image)
            t, x, theta = times[i], xs[i], float(r["theta_rad_unwrapped"])
            success = t + 1e-10 >= ts
            in_region = r["in_region"] == "True"
            color = TEAL if success else ORANGE
            cx = px(x)
            tip = (cx + scale * math.sin(theta), py - scale * math.cos(theta))
            draw.rectangle((cx-38, py+7, cx+38, py+45), fill="#d6dfdf", outline=INK, width=2)
            for wheelx in (cx-24, cx+24):
                draw.ellipse((wheelx-8, py+45, wheelx+8, py+61), fill=INK)
            draw.line(((cx, py), tip), fill=INK, width=9)
            draw.ellipse((tip[0]-7, tip[1]-7, tip[0]+7, tip[1]+7), fill=TEAL)
            draw.ellipse((cx-7, py-7, cx+7, py+7), fill=ORANGE, outline=INK, width=1)
            force = forces[i]
            if abs(force) > .05:
                fx = cx + force / 10 * 85
                fy = py+26
                draw.line((cx, fy, fx, fy), fill=ORANGE, width=4)
                direction = 1 if force > 0 else -1
                draw.polygon([(fx, fy), (fx-direction*10, fy-5), (fx-direction*10, fy+5)], fill=ORANGE)
            draw.text((919, 113), "PHYSICAL EVALUATION TIME", font=fs[13], fill=MUTED)
            draw.text((915, 136), f"{t:05.2f} s", font=fs[40], fill=INK)
            draw.text((919, 198), "FIRST 2 s HOLD COMPLETED" if success else "SEEKING THE FIRST 2 s HOLD", font=fs[14], fill=color)
            draw.text((919, 225), f"T_s = {ts:.3f} s" if success else "Swing up, center and settle", font=fs[18], fill=color)
            draw.text((919, 266), "All four limits apply simultaneously", font=fs[13], fill=MUTED)
            values = [("x", x, .1, "m"), ("v", float(r["v_m_s"]), .1, "m/s"),
                      ("angle", angles[i], 5, "deg"), ("omega", float(r["omega_rad_s"]), .2, "rad/s")]
            for j, (name, value, limit, unit) in enumerate(values):
                y = 296 + j*39
                draw.text((919, y), f"{name}: {value:+.3f} {unit}", font=fs[16], fill=TEAL if abs(value)<=limit+1e-12 else INK)
                draw.text((919, y+20), f"limit +/-{limit:g} {unit}", font=fs[13], fill=MUTED)
            draw.text((919, 465), "Inside tolerance region" if in_region else "Outside tolerance region", font=fs[14], fill=TEAL if in_region else MUTED)
            for left, right, top, bottom, points, line_color in panel_geometries:
                cursor = left + t / 20 * (right-left)
                draw.line((cursor, top, cursor, bottom), fill=line_color, width=2)
                qx, qy = points[i]
                draw.ellipse((qx-3, qy-3, qx+3, qy+3), fill=line_color)
            if i in (0, math.ceil(ts/.02), 1000):
                image.save(BASE / f"preview_{i:04d}.png")
            if i == 0:
                image.save(BASE / "preview.png")
            process.stdin.write(image.tobytes())
            if i % 250 == 0:
                print(f"Rendered {i}/1000 control samples", flush=True)
        process.stdin.close()
        error = process.stderr.read().decode()
        assert process.wait() == 0, error
    print(json.dumps({"output": str(output), "frames": len(rows), "fps": FPS,
                      "video_duration_s": len(rows)/FPS, "physical_time_span_s": 20}))


if __name__ == "__main__":
    render()
