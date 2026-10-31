
import argparse
import glob
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import intref  # noqa: E402
from export_rtl import IN_SHAPE, expected_cycles, geometry  # noqa: E402

CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']
REPLY = struct.Struct('<BB8iI')   # 0x5A, mask, 8 x int32 logits, uint32 compute cycles = 38 bytes
CLK_HZ = 27e6


def to_planes(x_uint8, factor=4):
    """(N,224,224) maps -> (N,2,56,56) uint8: per 4x4 block, how many dies are on the wafer and how many failed."""
    n, h, w = x_uint8.shape
    planes = np.stack([x_uint8 > 0, x_uint8 == 254], 1).astype(np.uint8)
    return planes.reshape(n, 2, h // factor, factor, w // factor, factor).sum((3, 5), dtype=np.uint8)


def find_ports():
    return sorted(glob.glob('/dev/cu.usbserial-*') + glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'))


def open_board(port):
    import serial
    if not port:
        ports = find_ports()
        if not ports:
            sys.exit('No USB serial port found. Is the Tang Nano 9K plugged in?')
        # the BL702 USB bridge exposes JTAG (first) and UART (second)
        port = ([p for p in ports if p.endswith('1')] or ports)[-1]
    return serial.Serial(port, 115200, timeout=5), port


def reload_params(link):
    """Command 0xA9: send the weight, parameter and head memories from rtl/gen/ instead of relying on the
    contents built into the bitstream (also lets a re-exported model of the same shape run without a rebuild)."""
    for sel, (name, nbytes) in enumerate([('wrom.hex', 1), ('prom.hex', 7), ('hrom.hex', 1)]):
        words = [int(t, 16) for t in (ROOT / 'rtl/gen' / name).read_text().split()]
        link.reset_input_buffer()
        link.write(bytes([0xA9, sel]) + b''.join(w.to_bytes(nbytes, 'little') for w in words))
        if link.read(2) != bytes([0x5E, sel]):
            raise RuntimeError(f'{name}: the board did not acknowledge the upload')


def classify(link, counts):
    """One wafer -> (mask, logits, compute cycles, round-trip seconds)."""
    link.reset_input_buffer()
    t0 = time.perf_counter()
    link.write(bytes([0xA5]) + counts.astype(np.uint8).tobytes())
    raw = link.read(REPLY.size)
    dt = time.perf_counter() - t0
    if len(raw) != REPLY.size or raw[0] != 0x5A:
        raise RuntimeError('no valid reply: check the port and that the bitstream is loaded')
    _, mask, *rest = REPLY.unpack(raw)
    return mask, np.array(rest[:8], np.int64), rest[8], dt


def load_test(path):
    if not path or not Path(path).exists():
        sys.exit('Test data not found: pass --data <dataset_15_TEST.npz> or set WAFER_TEST_NPZ.')
    d = np.load(path)
    return d['x'], d['y']


def ascii_map(c):
    wafer, fail = c[0].astype(int), c[1].astype(int)
    rows = []
    for r in range(0, 56, 2):
        row = ''
        for q in range(0, 56, 2):
            w, f = wafer[r:r + 2, q:q + 2].sum(), fail[r:r + 2, q:q + 2].sum()
            row += ' ' if w == 0 else ('#' if f > w * 0.5 else '+' if f > w * 0.15 else '.')
        rows.append(row)
    return '\n'.join(rows)


def names(bits):
    return ', '.join(CLASSES[k] for k in range(8) if bits[k]) or 'none'


def cmd_demo(a, layers, head):
    x, y = load_test(a.data)
    c = to_planes(x[a.idx:a.idx + 1])[0]
    link, port = open_board(a.port)
    if a.reload:
        reload_params(link)
    mask, logits, cycles, dt = classify(link, c)
    link.close()
    print(ascii_map(c))
    print(f'\ntest wafer {a.idx} on {port}   (# mostly failing, + some failures, . passing die)')
    print(f'ground truth  : {names(y[a.idx] > 0)}')
    print(f'FPGA predicts : {names([(mask >> k) & 1 for k in range(8)])}')
    print(f'scores        : {logits.tolist()}   (defect when > 0)')
    print(f'same as the software model: {np.array_equal(logits, intref.run(layers, head, c[None])[0])}   '
          f'compute {cycles / CLK_HZ * 1e3:.1f} ms ({cycles} clocks)   round trip {dt * 1e3:.0f} ms')


def cmd_run(a, layers, head):
    x, y = load_test(a.data)
    ids = np.random.default_rng(a.seed).choice(len(x), a.n, replace=False)
    counts = to_planes(x[ids])
    reference = intref.run(layers, head, counts)
    link, port = open_board(a.port)
    if a.reload:
        reload_params(link)
    pred, same, times = np.zeros((a.n, 8), bool), 0, []
    for i, c in enumerate(counts):
        _, logits, _, dt = classify(link, c)
        pred[i] = logits > 0
        same += np.array_equal(logits, reference[i])
        times.append(dt)
    link.close()
    truth = y[ids] > 0
    tp, fp, fn = (pred & truth).sum(0), (pred & ~truth).sum(0), (~pred & truth).sum(0)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    print(f'{port}: {a.n} random test wafers')
    print(f'  identical to the software model: {same}/{a.n}')
    print(f'  macro F1 of the FPGA on these wafers: {f1.mean():.4f}')
    print(f'  round trip {np.mean(times) * 1e3:.0f} ms per wafer (compute '
          f'{expected_cycles(geometry(layers, *IN_SHAPE)) / CLK_HZ * 1e3:.1f} ms, the rest is USB transfer)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('ports')
    for name in ('demo', 'run'):
        s = sub.add_parser(name)
        s.add_argument('--data', default=os.environ.get('WAFER_TEST_NPZ'), help='test split .npz (x, y)')
        s.add_argument('--model', default=str(ROOT / 'model/wafer_cnn_int8.npz'))
        s.add_argument('--port', help='serial port (default: auto-detect)')
        s.add_argument('--reload', action='store_true', help='upload the weights from rtl/gen/ first')
        if name == 'demo':
            s.add_argument('--idx', type=int, default=2119, help='index into the test split')
        else:
            s.add_argument('--n', type=int, default=200)
            s.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.cmd == 'ports':
        print('\n'.join(find_ports()) or 'no USB serial ports found')
        return
    layers, head = intref.load(a.model)
    (cmd_demo if a.cmd == 'demo' else cmd_run)(a, layers, head)


if __name__ == '__main__':
    main()
