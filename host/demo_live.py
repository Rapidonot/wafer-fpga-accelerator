"""Live demo for filming: the Mac shows each wafer, the Tang Nano 9K classifies it, and the board's LEDs light up.

LEDs: 1 busy (receiving, computing, replying), 2 result ready, 3-6 the Center, Donut, Edge-Loc and Edge-Ring
outputs of the last wafer. Every reply is also checked against the integer software model.

  python host/demo_live.py                 # 11 wafers, ~45 s
  python host/demo_live.py --hold 4        # seconds each result stays on screen
  python host/demo_live.py --save DIR      # also save each screen as a PNG

The wafers are one random test wafer per defect class, two multi-defect wafers and one defect-free wafer, drawn
with a fixed seed. They were not chosen by whether the model gets them right.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from wafer_host import CLASSES, CLK_HZ, ROOT, classify, intref, load_test, open_board, to_planes

DEMO_IDS = [7518, 4334, 9988, 4110, 9882, 2120, 506, 5551, 296, 4865, 7525]
BG, FG, DIM, GOOD, BAD, ACCENT = '#0f1115', '#e8e8e8', '#8a8f98', '#3ddc84', '#ff5c5c', '#5aa9ff'


def names(bits):
    return ', '.join(CLASSES[k] for k in range(8) if bits[k]) or 'none (good wafer)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('WAFER_TEST_NPZ'), help='test split .npz (x, y)')
    ap.add_argument('--model', default=str(ROOT / 'model/wafer_cnn_int8.npz'))
    ap.add_argument('--hold', type=float, default=3.0)
    ap.add_argument('--port')
    ap.add_argument('--save', help='folder to save a PNG of each result screen')
    a = ap.parse_args()

    import matplotlib
    if a.save and not sys.stdout.isatty():
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    layers, head = intref.load(a.model)
    x, y = load_test(a.data)
    link, port = open_board(a.port)
    cmap = ListedColormap(['#1b1b2f', '#2a9d8f', '#f4d35e'])  # off-wafer, pass, fail

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': FG})
    fig = plt.figure(figsize=(14, 7.9), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title('Wafer defect classifier on a Tang Nano 9K FPGA')
    except Exception:
        pass
    ax_img = fig.add_axes([0.04, 0.12, 0.42, 0.76])
    ax_txt = fig.add_axes([0.50, 0.08, 0.48, 0.84])
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)

    def draw(i, j, state, pred=None, cycles=None, exact=None, tally=(0, 0)):
        ax_img.clear()
        ax_img.imshow(x[j], cmap=cmap, vmin=0, vmax=254, interpolation='nearest')
        ax_img.set_axis_off()
        ax_img.set_title(f'Test wafer #{j}', color=DIM, fontsize=13, pad=10)
        ax_txt.clear()
        ax_txt.set_axis_off()
        ax_txt.set_facecolor(BG)
        t = ax_txt.text
        t(0, 0.97, 'Wafer defect classifier', fontsize=26, weight='bold', va='top')
        t(0, 0.89, 'running on a custom accelerator on a Tang Nano 9K FPGA', fontsize=16, color=ACCENT, va='top')
        t(0, 0.78, f'Wafer {i + 1} of {len(DEMO_IDS)}', fontsize=12, color=DIM, va='top')
        t(0, 0.70, 'True defects', fontsize=12, color=DIM, va='top')
        t(0, 0.65, names(y[j]), fontsize=19, va='top')
        t(0, 0.53, 'FPGA predicts', fontsize=12, color=DIM, va='top')
        if state == 'computing':
            t(0, 0.48, 'classifying on the board...', fontsize=19, color=ACCENT, va='top')
        else:
            colour = GOOD if exact else BAD
            t(0, 0.48, names(pred) + ('   ✓' if exact else '   ✗'), fontsize=19, color=colour, va='top')
            t(0, 0.36, f'Computed on the FPGA in {cycles / CLK_HZ * 1e3:.1f} ms ({cycles:,} clocks at 27 MHz)',
              fontsize=14, va='top')
            t(0, 0.30, '8 outputs identical to the Python model (bit-exact) ✓', fontsize=14, color=GOOD,
              va='top')
            t(0, 0.21, f'This run: {tally[0]} / {tally[1]} wafers fully correct', fontsize=14, va='top')
        t(0, 0.04, 'Board LEDs 3-6: Center, Donut, Edge-Loc, Edge-Ring', fontsize=11, color=DIM, va='bottom')
        fig.canvas.draw_idle()
        plt.pause(0.05)

    plt.ion()
    plt.show(block=False)
    correct, done = 0, 0
    try:
        for i, j in enumerate(DEMO_IDS):
            draw(i, j, 'computing')
            time.sleep(0.6)  # let the viewer see the wafer before the board answers
            counts = to_planes(x[j:j + 1])[0]
            _, logits, cycles, _ = classify(link, counts)
            golden = intref.run(layers, head, counts[None])[0]
            if not np.array_equal(logits, golden):
                raise RuntimeError(f'wafer {j}: board output differs from the Python model')
            pred = logits > 0
            exact = bool(np.array_equal(pred, y[j].astype(bool)))
            done += 1
            correct += exact
            draw(i, j, 'result', pred, cycles, exact, (correct, done))
            if a.save:
                fig.savefig(Path(a.save) / f'demo_{i + 1:02d}_wafer{j}.png', facecolor=BG, dpi=110)
            plt.pause(a.hold)
    finally:
        link.close()
    print(f'{port}: {correct}/{done} wafers fully correct; every board output matched the Python model')
    plt.ioff()
    if matplotlib.get_backend().lower() != 'agg':
        print('Close the window to finish.')
        plt.show()  # keep the last screen up until the window is closed


if __name__ == '__main__':
    main()
