"""Integer reference model of the accelerator: the exact arithmetic the RTL implements, in plain numpy."""

import numpy as np


def load(npz_path):
    """Integer model file -> (layers, head)."""
    z = np.load(npz_path)
    layers = [{k: z[f'l{i}_{k}'] for k in ('w', 'b', 'M', 'S')} for i in range(int(z['n_layers']))]
    for i, L in enumerate(layers):
        L['pool'] = int(z[f'l{i}_pool']) if f'l{i}_pool' in z.files else 1
    return layers, {'w': z['head_w'], 'b': z['head_b']}


def conv3x3_acc(a, w, b):
    """a: (N,C,H,W) int, w: (O,C,3,3) int8, b: (O,) -> (N,O,H,W) int64 accumulators, zero padding."""
    n, c, h, wd = a.shape
    p = np.pad(a.astype(np.int64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    acc = np.zeros((n, w.shape[0], h, wd), np.int64)
    w64 = w.astype(np.int64)
    for dy in range(3):
        for dx in range(3):
            acc += np.einsum('nchw,oc->nohw', p[:, :, dy:dy + h, dx:dx + wd], w64[:, :, dy, dx], optimize=True)
    return acc + b.astype(np.int64).reshape(1, -1, 1, 1)


def pool2_max(acc):
    """2x2/stride-2 max pool; an odd last row/column is dropped."""
    n, o, h, w = acc.shape
    acc = acc[:, :, :h // 2 * 2, :w // 2 * 2]
    return acc.reshape(n, o, h // 2, 2, w // 2, 2).max((3, 5))


def requant(acc, m, s):
    m = m.astype(np.int64).reshape(1, -1, 1, 1)
    s = s.astype(np.int64).reshape(1, -1, 1, 1)
    return np.clip((acc * m + (np.int64(1) << (s - 1))) >> s, 0, 255)


def run(layers, head, counts):
    """counts: (N,2,56,56) uint8 block counts -> (N,8) int64 logits."""
    a = counts.astype(np.int64)
    for L in layers:
        acc = conv3x3_acc(a, L['w'], L['b'])
        if L['pool']:
            acc = pool2_max(acc)
        a = requant(acc, L['M'], L['S'])
    g = a.sum((2, 3))
    return g @ head['w'].astype(np.int64).T + head['b'].astype(np.int64)
