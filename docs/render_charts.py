"""Render README charts from recorded evidence; requires matplotlib.

From the repository root: uv run --with matplotlib python docs/render_charts.py
This does not load model weights or run training.
"""
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets'
BG, FG, MUTED, GRID = '#0f172a', '#f1f5f9', '#cbd5e1', '#334155'
BLUE, RED, TEAL = '#93c5fd', '#fda4af', '#5eead4'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12,
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'text.color': FG, 'axes.labelcolor': MUTED,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'savefig.facecolor': BG,
})


def read(path):
    return json.loads((ROOT / path).read_text())


def clean(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def badges():
    # Local SVGs keep GitHub badges independent of third-party image services.
    entries = [
        ('accuracy', 'Test accuracy', '88.89%', '#0f766e'),
        ('gain', 'vs base', '+18.18 pp', '#1d4ed8'),
        ('samples', 'Training examples', '1,080', '#475569'),
        ('runtime', 'Apple Silicon', 'MLX', '#475569'),
        ('experiment', 'Experiment', 'Local QLoRA', '#475569'),
        ('correct', 'vs base', '+54 correct', '#0f766e'),
    ]
    for name, label, value, color in entries:
        left = len(label) * 7 + 20
        right = len(value) * 7 + 20
        width = left + right
        title = escape(f'{label}: {value}')
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="28" viewBox="0 0 {width} 28" role="img" aria-label="{title}">
<title>{title}</title>
<clipPath id="round"><rect width="{width}" height="28" rx="4"/></clipPath>
<g clip-path="url(#round)"><rect width="{left}" height="28" fill="#334155"/><rect x="{left}" width="{right}" height="28" fill="{color}"/></g>
<g fill="#fff" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11" text-anchor="middle">
<text x="{left/2}" y="18">{escape(label)}</text><text x="{left+right/2}" y="18" font-weight="bold">{escape(value)}</text>
</g></svg>
'''
        (OUT / f'badge-{name}.svg').write_text(svg)


def results():
    sources = [
        'results/20261028T002422930075Z.json',
        'results/20261028T004036824182Z.json',
        'results/20261028T115822561383Z.json',
    ]
    summaries = [read(p)['summary'] for p in sources]
    total = summaries[0]['intent_cases']
    assert total == 297 and all(s['intent_cases'] == total for s in summaries)
    counts = [s['correct_intent_and_schema'] for s in summaries]
    scores = [100 * n / total for n in counts]
    fig, ax = plt.subplots(figsize=(12, 5.9), dpi=180)
    fig.subplots_adjust(left=.29, right=.93, top=.70, bottom=.23)
    fig.text(.055, .90, 'Same model. Same test. A better second run.', fontsize=22, weight='bold')
    fig.text(.055, .82, 'Dolphin 8B  /  297 customer support classifications  /  local QLoRA', color=MUTED, fontsize=12)
    ys = [2, 1, 0]
    ax.barh(ys, [100] * 3, height=.46, color='#1e293b')
    ax.barh(ys, scores, height=.46, color=[BLUE, RED, TEAL])
    labels = ['BASE MODEL\nBefore fine-tuning', 'RUN 001\nLR 0.0001 · step 540', 'RUN 002\nLR 0.00001 · selected step 450']
    ax.set_yticks(ys, labels, fontsize=11, ha='right')
    ax.tick_params(axis='y', pad=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(-.6, 2.6)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis='x', color=GRID, alpha=.5)
    clean(ax)
    for y, score, count in zip(ys, scores, counts):
        ax.text(score - 2, y, f'{score:.2f}%', ha='right', va='center', color=BG, fontsize=16, weight='bold')
        ax.text(101.5, y, f'{count}/{total}', va='center', color=MUTED, fontsize=11)
    fig.text(.055, .082, f'+{scores[2]-scores[0]:.2f} percentage points over the base', color=TEAL, weight='bold', fontsize=15)
    fig.text(.055, .035, 'Run 002 checkpoint chosen using validation data. One seed; previously used final suite.', color=MUTED, fontsize=10)
    fig.savefig(OUT / 'results.png')
    plt.close(fig)


def validation():
    path = 'experiments/training-002-lr1e5/'
    summary = read(path + 'summary.json')
    checks = read(path + 'monitor.json')
    steps = [0] + [r['completed_optimizer_steps'] for r in checks]
    correct = [summary['base_monitored']['correct']] + [r['correct'] for r in checks]
    total = summary['base_monitored']['total']
    selected = summary['selected_checkpoint']['completed_optimizer_steps']
    selected_count = correct[steps.index(selected)]
    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=180)
    fig.subplots_adjust(left=.09, right=.95, top=.72, bottom=.23)
    fig.text(.055, .90, 'Check the answers while training.', fontsize=23, weight='bold')
    fig.text(.055, .82, f'Run 002  /  fixed {total}-question validation subset  /  learning rate 0.00001', color=MUTED, fontsize=12)
    ax.plot(steps, correct, color=TEAL, linewidth=2.5, marker='o', markersize=6)
    ax.fill_between(steps, correct, color=TEAL, alpha=.07)
    ax.axvline(selected, color=TEAL, linestyle='--', alpha=.45)
    ax.scatter([selected], [selected_count], s=140, facecolor=TEAL, edgecolor=FG, linewidth=1.6, zorder=4)
    ax.annotate(f'Selected: step {selected}\n{selected_count}/{total} correct', xy=(selected, selected_count),
                xytext=(275, 16), fontsize=12, weight='bold', color=TEAL,
                arrowprops={'arrowstyle': '->', 'color': TEAL, 'connectionstyle': 'arc3,rad=.15'})
    ax.text(10, 15.2, f'Base: {correct[0]}/{total}', color=MUTED, fontsize=11)
    ax.set_xlim(0, 555)
    ax.set_ylim(0, total + 1)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 540])
    ax.set_yticks([0, 9, 18, 27])
    ax.set_xlabel('Completed training updates', labelpad=10)
    ax.set_ylabel(f'Correct answers / {total}', labelpad=10)
    ax.grid(axis='y', color=GRID, alpha=.5)
    clean(ax)
    fig.text(.055, .075, 'Train 50 updates → check 27 questions → resume training', fontsize=14, weight='bold')
    fig.text(.055, .03, 'Validation questions were not used for weight updates. Later tied scores kept the earlier checkpoint.', color=MUTED, fontsize=10)
    fig.savefig(OUT / 'validation.png')
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    badges()
    results()
    validation()
    print('Rendered local badges and both charts in docs/assets/ from recorded evidence.')
