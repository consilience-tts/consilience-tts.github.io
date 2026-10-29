import re, sys, xml.etree.ElementTree as ET
NS = '{http://www.w3.org/2000/svg}'
def parse_d(d):
    toks = re.findall(r'[MLCZHVmlczhv]|-?\d*\.?\d+(?:e-?\d+)?', d)
    pts, cmd, i = [], None, 0
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t; i += 1
            if cmd in 'Zz': continue
            continue
        if cmd in 'ML':
            pts.append((float(toks[i]), float(toks[i+1]), cmd)); i += 2
        elif cmd == 'C':
            pts.append((float(toks[i+4]), float(toks[i+5]), 'C')); i += 6
        else:
            i += 1
    return pts
def mat(tr):
    if not tr: return (1,0,0,1,0,0)
    m = re.match(r'matrix\(([^)]*)\)', tr)
    return tuple(float(v) for v in re.split(r'[ ,]+', m.group(1).strip()))
def apply(m, x, y):
    a,b,c,d,e,f = m
    return (a*x + c*y + e, b*x + d*y + f)
def paths(fn):
    root = ET.parse(fn).getroot()
    out = []
    for el in root.iter(NS+'path'):
        # skip glyph defs
        pass
    def walk(node, inherited, in_defs):
        for ch in node:
            tag = ch.tag.replace(NS,'')
            if tag in ('defs','clipPath','symbol') : continue
            if tag == 'path':
                st = {}
                for k in ('fill','stroke','stroke-width','fill-opacity','stroke-opacity','stroke-dasharray'):
                    if ch.get(k) is not None: st[k] = ch.get(k)
                m = mat(ch.get('transform'))
                pts = [apply(m, x, y)+(c,) for x,y,c in parse_d(ch.get('d',''))]
                out.append((st, pts))
            walk(ch, inherited, in_defs)
    walk(root, {}, False)
    return out
if __name__ == '__main__':
    for i,(st,pts) in enumerate(paths(sys.argv[1])):
        if len(pts) >= int(sys.argv[2]) if len(sys.argv)>2 else 1:
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            print(i, st, len(pts), 'x[%.1f,%.1f] y[%.1f,%.1f]'%(min(xs),max(xs),min(ys),max(ys)))
