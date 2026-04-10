#!/usr/bin/env python3
"""Regenerate key paper figures with improved aesthetics."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

NUS_BLUE = '#003D7C'
SAFE_GREEN = '#2E8B57'
WARN_ORANGE = '#E67E22'
FAIL_RED = '#C0392B'
LIGHT_BG = '#F8F9FA'

OUT = '/home/kaixin/yisong/3ddl/docs/figures'

# ═══════════════════════════════════════════════════════════════════
# Fig 4.4: Pruning Pareto Front (improved)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.set_facecolor(LIGHT_BG)

# Data: ratio, latency_ms, avg_dice, c3, c4, params_M
data = [
    ('Teacher',  17.40, 0.9054, 0.7937, 0.8966, 5.75),
    ('r=0.375',  15.28, 0.8951, 0.7869, 0.8717, 2.25),
    ('r=0.5',     6.39, 0.8895, 0.7545, 0.8749, 1.44),
    ('r=0.625',   5.00, 0.8821, 0.7488, 0.8618, 0.81),
]

colors = [FAIL_RED, NUS_BLUE, WARN_ORANGE, SAFE_GREEN]
markers = ['*', 'o', 's', 'D']

for i, (label, lat, dice, c3, c4, params) in enumerate(data):
    size = params * 60
    ax.scatter(lat, dice, s=size, c=colors[i], marker=markers[i],
               edgecolors='white', linewidth=1.5, zorder=5)
    # Annotation with c3/c4 info
    offset = (12, 8) if label != 'r=0.375' else (12, -15)
    ax.annotate(f'{label}\nc3={c3:.3f}  c4={c4:.3f}',
                (lat, dice), textcoords='offset points', xytext=offset,
                fontsize=9, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

# Pareto front line
pareto_x = [17.40, 15.28, 6.39, 5.00]
pareto_y = [0.9054, 0.8951, 0.8895, 0.8821]
ax.plot(pareto_x, pareto_y, '--', color='gray', alpha=0.5, zorder=1)

ax.set_xlabel('Patch Inference Latency (ms)')
ax.set_ylabel('Full-Case Average Dice')
ax.set_title('Pruning-Ratio Pareto Front', fontweight='bold')

# Size legend
for p, label in [(5.75, '5.75M'), (1.44, '1.44M')]:
    ax.scatter([], [], s=p*60, c='gray', alpha=0.5, label=f'{label} params')
ax.legend(loc='lower left', framealpha=0.9)

plt.savefig(f'{OUT}/pruning_ratio_pareto.png')
plt.close()
print('Saved: pruning_ratio_pareto.png')


# ═══════════════════════════════════════════════════════════════════
# Fig 4.5: Deployment Gap (improved — more data points, clearer message)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.set_facecolor(LIGHT_BG)

# Data: label, fc_dice, total_flips(pad+solder), gate2
variants = [
    ('Teacher\n(eager)',          0.9054,   0, True),
    ('Teacher\nTRT FP16',        0.9054,   7, False),
    ('Teacher\nTRT INT8',        0.9054,   7, False),
    ('r=0.375\neager',           0.8951, 169, False),
    ('r=0.375\nTRT FP16',       0.8951, 169, False),
    ('r=0.5\neager',             0.8895, 210, False),
    ('r=0.5\nTRT FP16',         0.8895, 212, False),
]

for label, dice, flips, safe in variants:
    color = SAFE_GREEN if safe else (WARN_ORANGE if flips < 20 else FAIL_RED)
    marker = 'o' if safe else ('D' if flips < 20 else 'X')
    ax.scatter(dice, flips, s=120, c=color, marker=marker,
               edgecolors='white', linewidth=1.5, zorder=5)
    ha = 'right' if dice > 0.90 else 'left'
    offset = (-10, 8) if dice > 0.90 else (10, 8)
    ax.annotate(label, (dice, flips), textcoords='offset points',
                xytext=offset, fontsize=8, ha=ha, va='bottom')

# Gate 2 boundary
ax.axhline(y=0, color=SAFE_GREEN, linewidth=2, linestyle='-', alpha=0.7)
ax.axhspan(-5, 0.5, color=SAFE_GREEN, alpha=0.08)
ax.text(0.887, 2, 'Gate 2 safe zone (zero flips)', fontsize=9,
        color=SAFE_GREEN, fontstyle='italic')

# Deployment gap arrow
ax.annotate('', xy=(0.8895, 210), xytext=(0.9054, 0),
            arrowprops=dict(arrowstyle='->', color=FAIL_RED, lw=2, ls='--'))
ax.text(0.897, 100, 'Deployment\nGap', fontsize=11, fontweight='bold',
        color=FAIL_RED, ha='center', rotation=0)

ax.set_xlabel('Full-Case Average Dice (174 test cases)')
ax.set_ylabel('Total Defect-Flag Flips (pad + solder)')
ax.set_title('Deployment Gap: High Dice $\\neq$ Safe Deployment', fontweight='bold')
ax.set_ylim(-10, 230)
ax.set_xlim(0.886, 0.908)

plt.savefig(f'{OUT}/deployment_gap_fullcase_vs_pipeline.png')
plt.close()
print('Saved: deployment_gap_fullcase_vs_pipeline.png')


# ═══════════════════════════════════════════════════════════════════
# Fig 4.6: Runtime-Quality Ranking (improved)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
ax.set_facecolor(LIGHT_BG)

# Data: label, total_s, wp_c3_dice, energy_J, gate2
candidates = [
    ('Teacher / eager',       34.96, 1.000, 4945, True),
    ('Teacher / TRT FP16',    42.09, 0.948, 3800, False),
    ('Teacher / TRT INT8',    39.19, 0.948, 3700, False),
    ('r=0.375 / eager',       33.38, 0.539, 4833, False),
    ('r=0.375 / TRT FP16',   30.41, 0.539, 3746, False),
    ('r=0.5 / eager',         30.39, 0.216, 3816, False),
    ('r=0.5 / TRT FP16',     29.45, 0.216, 3499, False),
]

for label, total, c3, energy, safe in candidates:
    size = energy / 15
    color = SAFE_GREEN if safe else (WARN_ORANGE if c3 > 0.9 else FAIL_RED)
    edgecolor = 'black' if safe else 'gray'
    lw = 2 if safe else 1
    ax.scatter(total, c3, s=size, c=color, edgecolors=edgecolor,
               linewidth=lw, zorder=5, alpha=0.85)
    # Smart label placement
    if 'r=0.5' in label and 'eager' in label:
        offset = (-5, -12)
    elif 'Teacher / eager' in label:
        offset = (10, -8)
    elif 'Teacher / TRT' in label:
        offset = (8, 5)
    else:
        offset = (8, -8)
    ax.annotate(label, (total, c3), textcoords='offset points',
                xytext=offset, fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# Gate 2 threshold line
ax.axhline(y=0.90, color=NUS_BLUE, linewidth=1.5, linestyle='--', alpha=0.6)
ax.text(28.5, 0.91, 'Gate 2: WP c3 $\\geq$ 0.90', fontsize=9, color=NUS_BLUE)

ax.set_xlabel('Whole-Pipeline Total Latency (s)')
ax.set_ylabel('Whole-Pipeline Class-3 Dice (vs. Teacher)')
ax.set_title('Runtime--Quality--Energy Ranking', fontweight='bold')

# Energy legend
for e, lab in [(5000, '5000 J'), (3000, '3000 J')]:
    ax.scatter([], [], s=e/15, c='gray', alpha=0.4, label=lab)
safe_patch = mpatches.Patch(color=SAFE_GREEN, alpha=0.7, label='Gate 2 Pass')
near_patch = mpatches.Patch(color=WARN_ORANGE, alpha=0.7, label='Near-pass')
fail_patch = mpatches.Patch(color=FAIL_RED, alpha=0.7, label='Gate 2 Fail')
ax.legend(handles=[safe_patch, near_patch, fail_patch,
          plt.scatter([], [], s=5000/15, c='gray', alpha=0.4),
          plt.scatter([], [], s=3000/15, c='gray', alpha=0.4)],
          labels=['Gate 2 Pass', 'Near-pass', 'Gate 2 Fail', '5000 J', '3000 J'],
          loc='lower left', framealpha=0.9, ncol=2, fontsize=8)

plt.savefig(f'{OUT}/runtime_candidate_ranking_latency_quality.png')
plt.close()
print('Saved: runtime_candidate_ranking_latency_quality.png')

print('\nAll figures regenerated.')
