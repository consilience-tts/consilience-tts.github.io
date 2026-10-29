import sys, json, re
sys.path.insert(0, '/tmp/cpage/tools')
from svgpaths import paths

def rgb(s):
    if not s or not s.startswith('rgb'): return None
    v = [float(x.strip().rstrip('%')) for x in s[4:-1].split(',')]
    return '#%02x%02x%02x' % tuple(round(x*2.55) for x in v)

def load(fn):
    out = []
    for st, pts in paths(fn):
        out.append(dict(stroke=rgb(st.get('stroke')), fill=rgb(st.get('fill')),
                        sw=float(st.get('stroke-width', 0)), dash=st.get('stroke-dasharray'),
                        fo=float(st.get('fill-opacity', 1)), so=float(st.get('stroke-opacity', 1)),
                        pts=[(p[0], p[1]) for p in pts]))
    return out

def ticks(P, frame):
    """frame=(x0,x1,y0,y1) in px (y0 top). Return x tick px list, y tick px list (black short marks)."""
    x0, x1, y0, y1 = frame
    xt, yt = [], []
    for p in P:
        if p['stroke'] != '#000000' or len(p['pts']) != 2: continue
        (ax, ay), (bx, by) = p['pts']
        if abs(ax-bx) < 0.01 and min(ay, by) >= y1-0.2 and max(ay, by) <= y1+5 and x0-0.5 <= ax <= x1+0.5 and abs(ay-by) > 1:
            xt.append(ax)
        if abs(ay-by) < 0.01 and max(ax, bx) <= x0+0.2 and min(ax, bx) >= x0-5 and y0-0.5 <= ay <= y1+0.5 and abs(ax-bx) > 1:
            yt.append(ay)
    return sorted(set(round(v, 3) for v in xt)), sorted(set(round(v, 3) for v in yt), reverse=True)

def lin(px, vals):
    """fit value = a*px + b from tick px positions and values."""
    n = len(px); assert n == len(vals) and n >= 2, (px, vals)
    mx = sum(px)/n; mv = sum(vals)/n
    a = sum((p-mx)*(v-mv) for p, v in zip(px, vals)) / sum((p-mx)**2 for p in px)
    b = mv - a*mx
    res = max(abs(a*p+b-v) for p, v in zip(px, vals))
    assert res < 1e-3*max(1, abs(max(vals))), ('bad fit', res)
    return lambda p: a*p + b

def frames(P):
    """axes frames = black spines: return list of (x0,x1,y0,y1) from bottom spines paired w/ left spines."""
    bottoms = []
    for p in P:
        if p['stroke'] == '#000000' and len(p['pts']) == 2 and p['sw'] == 0.8:
            (ax, ay), (bx, by) = p['pts']
            if abs(ay-by) < 0.01 and abs(ax-bx) > 50: bottoms.append((min(ax, bx), max(ax, bx), ay))
    lefts = []
    for p in P:
        if p['stroke'] == '#000000' and len(p['pts']) == 2 and p['sw'] == 0.8:
            (ax, ay), (bx, by) = p['pts']
            if abs(ax-bx) < 0.01 and abs(ay-by) > 50: lefts.append((ax, min(ay, by), max(ay, by)))
    fr = []
    for (x0, x1, y) in bottoms:
        for (lx, ly0, ly1) in lefts:
            if abs(lx-x0) < 0.5 and abs(ly1-y) < 0.5:
                fr.append((x0, x1, ly0, ly1))
    return sorted(set((round(a,3),round(b,3),round(c,3),round(d,3)) for a,b,c,d in fr))
