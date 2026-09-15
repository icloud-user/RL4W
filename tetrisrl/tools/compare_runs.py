"""Compare training runs from their logs.

Reads ``*.log`` files produced by ``python -m tetrisrl train`` (one per run) and
tabulates the learning curves side by side, so an A/B is decided from numbers
rather than impressions.

Usage:
    python tools/compare_runs.py rw_v1.log rw_small.log rw_mc.log
    python tools/compare_runs.py --dir . --pattern "rw_*.log"
"""
import argparse
import glob
import os
import re
import sys

ROUND_RE = re.compile(
    r'^round\s+(\d+)/(\d+)\s+\|\s+lines\s+([-\d.]+)\s+\|\s+score\s+([-\d.]+)\s+'
    r'\|\s+eps\s+([\d.]+)\s+\|\s+buf\s+(\d+)\s+\|\s+loss\s+(\S+)\s+\|\s+\+\s*(\d+)')
EVAL_RE = re.compile(r'^\s+eval:\s+([\d.]+) lines/game \(max (\d+)\), score (\d+), '
                     r'pieces (\d+)')


def parse(path):
    rounds, evals = [], []
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            m = ROUND_RE.match(line.rstrip())
            if m:
                rounds.append({
                    'round': int(m.group(1)),
                    'lines': float(m.group(3)),
                    'score': float(m.group(4)),
                    'eps': float(m.group(5)),
                    'buf': int(m.group(6)),
                    'transitions': int(m.group(8)),
                })
                continue
            m = EVAL_RE.match(line.rstrip())
            if m:
                evals.append({
                    'lines': float(m.group(1)),
                    'max': int(m.group(2)),
                    'score': int(m.group(3)),
                    'pieces': int(m.group(4)),
                })
    return rounds, evals


def rolling(values, window):
    """Trailing mean, so a noisy per-round series is readable."""
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i + 1 - lo))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='*')
    parser.add_argument('--dir', default='.')
    parser.add_argument('--pattern', default='rw_*.log')
    parser.add_argument('--window', type=int, default=20)
    args = parser.parse_args(argv)

    paths = args.logs or sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        print('no logs found')
        return 1

    data = {}
    for p in paths:
        rounds, evals = parse(p)
        if rounds:
            data[os.path.basename(p)] = (rounds, evals)

    if not data:
        print('no parsable runs')
        return 1

    print(f'trailing {args.window}-round mean of lines/game, by round\n')
    header = f'{"round":>6s}'
    for name in data:
        header += f'{name[:16]:>18s}'
    print(header)
    print('-' * len(header))

    # Progress grid: every 10 rounds, trailing mean per run.
    smoothed = {n: rolling([r['lines'] for r in v[0]], args.window)
                for n, v in data.items()}
    max_round = max(len(v[0]) for v in data.values())
    for r in range(9, max_round, 10):
        row = f'{r + 1:>6d}'
        for name in data:
            s = smoothed[name]
            row += f'{s[r]:>18.3f}' if r < len(s) else f'{"-":>18s}'
        print(row)

    print()
    print(f'{"run":<20s}{"rounds":>8s}{"best eval":>11s}{"its max":>9s}'
          f'{"final eval":>12s}{"last lines":>12s}{"eps":>7s}{"buf":>9s}')
    print('-' * 88)
    for name, (rounds, evals) in data.items():
        best = max((e['lines'] for e in evals), default=float('nan'))
        best_max = max((e['max'] for e in evals), default=0)
        final = evals[-1]['lines'] if evals else float('nan')
        tail = sum(r['lines'] for r in rounds[-20:]) / min(20, len(rounds))
        print(f'{name[:19]:<20s}{len(rounds):>8d}{best:>11.2f}{best_max:>9d}'
              f'{final:>12.2f}{tail:>12.3f}{rounds[-1]["eps"]:>7.3f}'
              f'{rounds[-1]["buf"]:>9d}')

    # Rank by best evaluation, which is what the checkpoint actually delivers.
    ranked = sorted(data.items(), key=lambda kv: -max(
        (e['lines'] for e in kv[1][1]), default=float('-inf')))
    print()
    print('ranked by best eval lines/game:')
    for i, (name, (_r, evals)) in enumerate(ranked, 1):
        best = max((e['lines'] for e in evals), default=float('nan'))
        print(f'  {i}. {name:<20s} {best:7.2f}'
              + ('   <- winner' if i == 1 else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
