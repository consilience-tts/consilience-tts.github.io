"""Re-extract the plotted data from the paper's vector figures into ../static/data.js.

Usage: python3 extract_figures.py /path/to/cons_colm/final_figure
Needs poppler's `pdftocairo` on PATH. Tick values below are read off each figure's axes;
the script fails loudly if the tick marks found in the SVG don't fit them.
"""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import load, frames, ticks, lin

FIG = sys.argv[1] if len(sys.argv) > 1 else '../../cons_colm/final_figure'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', 'data.js')
SRC = {
    'case': 'case_study_p3_qwen.pdf',
    'alpha': 'alpha_sweep_lcb-qwen_lcb-gptoss_0.30_hmmt-gptoss20b_gpqa-gptoss20b.pdf',
    'confdiff': 'conf_diff_qnorm_hard50_lcb-gptoss_lcb-qwen_hmmt-gptoss20b_gpqa-qwen.pdf',
    'hist_all_mean': 'hist_conf_all_mean_gptoss_lcb.pdf',
    'hist_hard_mean': 'hist_conf_hard30_mean_gptoss_lcb.pdf',
    'hist_hard_cons': 'hist_conf_hard30_consilience_a3_gptoss_lcb.pdf',
    'scaling': 'size_scaling_a3.0_lcb-qwen_lcb-gptoss20b_hmmt-gptoss20b_gpqa-gptoss20b.pdf',
}
tmp = tempfile.mkdtemp()
P = {}
for k, f in SRC.items():
    svg = os.path.join(tmp, k + '.svg')
    subprocess.run(['pdftocairo', '-svg', os.path.join(FIG, f), svg], check=True)
    P[k] = load(svg)

def inside(p, fr, tol=1.0):
    xs = [q[0] for q in p['pts']]; ys = [q[1] for q in p['pts']]
    return min(xs) >= fr[0]-tol and max(xs) <= fr[1]+tol and min(ys) >= fr[2]-tol and max(ys) <= fr[3]+tol

def maps(Pk, fr, xv, yv):
    xt, yt = ticks(Pk, fr)
    return lin(xt, xv), lin(yt, yv)

def interp(seg, x):
    for j in range(1, len(seg)):
        if seg[j-1][0] <= x <= seg[j][0] and seg[j][0] > seg[j-1][0]:
            (x0, y0), (x1, y1) = seg[j-1], seg[j]
            return y0 + (y1-y0)*(x-x0)/(x1-x0)
    return min(seg, key=lambda p: abs(p[0]-x))[1]

out = {}

# Case study: two smoothed per-token confidence traces (first 2048 think tokens).
Pk = P['case']; fr = frames(Pk)[0]
fx, fy = maps(Pk, fr, list(range(0, 2001, 250)), [4, 6, 8, 10, 12, 14])
tr = {}
for p in Pk:
    if p['sw'] == 2.0 and len(p['pts']) > 100:
        tr['cons' if p['stroke'] == '#2166ac' else 'mean'] = sorted((fx(x), fy(y)) for x, y in p['pts'])
grid = list(range(0, 2048, 2))
out['case'] = {'step': 2, **{k: [round(interp(v, x), 2) for x in grid] for k, v in tr.items()}}

# Alpha sweep (20% window variant) + pass@1 reference.
Pk = P['alpha']
ys = [[52, 54, 56, 58, 60], [66, 67, 68, 69, 70, 71], [70, 72.5, 75, 77.5, 80, 82.5], [71, 71.5, 72, 72.5, 73, 73.5]]
names = ['LiveCodeBench · Qwen3-80B', 'LiveCodeBench · GPT-OSS-120B', 'HMMT 2025 · GPT-OSS-20B', 'GPQA Diamond · GPT-OSS-20B']
out['alpha'] = []
for fr, yv, nm in zip(frames(Pk), ys, names):
    fx, fy = maps(Pk, fr, list(range(9)), yv)
    d = {'name': nm}
    for p in Pk:
        if p['sw'] != 2.0: continue
        if p['stroke'] == '#c0392b' and len(p['pts']) > 5 and inside(p, fr):
            d['pct'] = [[round(fx(x), 2), round(fy(y), 2)] for x, y in p['pts']]
        if p['stroke'] == '#808080' and len(p['pts']) == 2 and p['pts'][1][0]-p['pts'][0][0] > 200 and fr[0]-1 <= p['pts'][0][0] <= fr[1]:
            d['pass1'] = round(fy(p['pts'][0][1]), 2)
    out['alpha'].append(d)

# Correct-minus-incorrect confidence trajectories with ±SEM band.
Pk = P['confdiff']
ys = [[-0.05, 0, 0.05, 0.10], [-0.15, -0.10, -0.05, 0, 0.05], [-0.2, -0.1, 0, 0.1], [-0.05, 0, 0.05]]
names = ['LiveCodeBench · GPT-OSS-120B', 'LiveCodeBench · Qwen3-80B', 'HMMT 2025 · GPT-OSS-20B', 'GPQA Diamond · Qwen3-80B']
g = [k/200 for k in range(201)]
out['confdiff'] = []
for fr, yv, nm in zip(frames(Pk), ys, names):
    fx, fy = maps(Pk, fr, [0, 0.2, 0.4, 0.6, 0.8, 1.0], yv)
    mean = band = None
    for p in Pk:
        if not (fr[0]-1 <= min(q[0] for q in p['pts']) <= fr[1]): continue
        if p['stroke'] == '#c0392b' and p['sw'] == 2.0 and len(p['pts']) > 5: mean = sorted((fx(x), fy(y)) for x, y in p['pts'])
        if p['fill'] == '#c0392b' and p['stroke'] is None and len(p['pts']) > 5: band = [(fx(x), fy(y)) for x, y in p['pts']]
    i = max(range(len(band)), key=lambda j: (band[j][0], -j))
    e1 = band[1:i+1] if abs(band[0][0]-band[1][0]) < 1e-9 else band[:i+1]
    e1, e2 = sorted(e1), sorted(band[i+1:])
    A = [interp(e1, x) for x in g]; B = [interp(e2, x) for x in g]
    out['confdiff'].append({'name': nm, 'x': g, 'mean': [round(interp(mean, x), 4) for x in g],
                            'lo': [round(min(a, b), 4) for a, b in zip(A, B)], 'hi': [round(max(a, b), 4) for a, b in zip(A, B)]})

# Histograms: shared bins, legend swatches dropped, renormalised to unit area.
def hist(Pk, xv, yv):
    fr = frames(Pk)[0]; fx, fy = maps(Pk, fr, xv, yv)
    bars = {'correct': [], 'incorrect': []}; mu = {}
    for p in Pk:
        if p['fill'] in ('#2166ac', '#d6604d') and len(p['pts']) == 5 and abs(p['fo']-0.55) < 0.01:
            xs = [q[0] for q in p['pts']]; yy = [q[1] for q in p['pts']]
            bars['correct' if p['fill'] == '#2166ac' else 'incorrect'].append((fx(min(xs)), fx(max(xs)), fy(min(yy))))
        if p['sw'] == 1.8 and p['dash'] and len(p['pts']) == 2:
            mu['mu_correct' if p['stroke'] == '#2166ac' else 'mu_incorrect'] = round(fx(p['pts'][0][0]), 3)
    allb = bars['correct'] + bars['incorrect']
    ws = sorted(b-a for a, b, h in allb); w = ws[len(ws)//2]
    keep = [t for t in allb if abs((t[1]-t[0])-w) < 0.01*w]
    lo = min(t[0] for t in keep); hi = max(t[1] for t in keep); nb = round((hi-lo)/w); w = (hi-lo)/nb
    o = {'lo': round(lo, 4), 'w': round(w, 5), **mu}
    for cls, bs in bars.items():
        arr = [0.0]*nb
        for a, b, h in bs:
            if abs((b-a)-w) < 0.01*w: arr[round((a-lo)/w)] = h
        area = sum(arr)*w
        o[cls] = [round(v/area, 4) for v in arr]
    return o
out['hist'] = {
    'all_mean': hist(P['hist_all_mean'], [6, 8, 10, 12, 14], [0, .05, .10, .15, .20, .25, .30]),
    'hard_mean': hist(P['hist_hard_mean'], list(range(6, 14)), [0, .1, .2, .3, .4, .5]),
    'hard_cons': hist(P['hist_hard_cons'], [-35, -30, -25, -20, -15, -10, -5], [0, .025, .05, .075, .1, .125, .15, .175]),
}

# Accuracy vs. number of samples.
Pk = P['scaling']
ys = [[52, 54, 56, 58], [78, 79, 80, 81], [70, 75, 80], [72, 73, 74]]
names = ['LiveCodeBench · Qwen3-80B', 'LiveCodeBench · GPT-OSS-20B', 'HMMT 2025 · GPT-OSS-20B', 'GPQA Diamond · GPT-OSS-20B']
cmap = {'#808080': 'pass1', '#e67e22': 'mean', '#2980b9': 'last20', '#c0392b': 'cons'}
out['scaling'] = []
for fr, yv, nm in zip(frames(Pk), ys, names):
    fx, fy = maps(Pk, fr, list(range(8)), yv)
    d = {'name': nm}
    for p in Pk:
        if p['sw'] == 2.0 and p['stroke'] in cmap and len(p['pts']) > 5 and inside(p, fr):
            d[cmap[p['stroke']]] = [[2**round(fx(x)), round(fy(y), 2)] for x, y in p['pts']]
    out['scaling'].append(d)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    f.write('// Generated by tools/extract_figures.py from the paper figures. Do not edit by hand.\n')
    f.write('window.DATA = ' + json.dumps(out, separators=(',', ':')) + ';\n')
print('wrote', os.path.normpath(OUT), os.path.getsize(OUT), 'bytes')
for h in out['hist'].values(): print('  hist mu', h['mu_correct'], h['mu_incorrect'])
for a in out['alpha']: print('  alpha', a['name'], 'pass@1', a['pass1'], 'max', max(v for _, v in a['pct']))
