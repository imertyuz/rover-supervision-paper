"""Trace-based validation of the analytical model (M2-M4).
Input: inv/T{1..4}.pkl produced by load.py from the analysis bags.
All statistics are restricted to armed-AUTO intervals derived from /mavros/state.
"""
import pickle, collections, statistics as st, math, json, sys

CLASSES = ('person', 'tree', 'pole', 'curb_step', 'vehicle', 'animal')

def auto_windows(state):
    W, s = [], None
    for t, m, a in state:
        on = (m == 'AUTO' and a)
        if on and s is None: s = t
        if not on and s is not None: W.append((s, t)); s = None
    if s is not None: W.append((s, state[-1][0]))
    return W

def inw(t, W): return any(a <= t <= b for a, b in W)

MISSION_S = {'T1': 284.4, 'T2': 291.5, 'T3': 317.0, 'T4': 463.7}  # Table tab:missions, first AUTO -> Mission Complete

def armed_auto_selector(state, T):
    """Manuscript rule: inside [first AUTO, Mission Complete]; preceding FCU state armed AUTO and <= 2 s old."""
    import bisect
    t0 = next(t for t, m, a in state if m == 'AUTO'); t1 = t0 + MISSION_S[T]
    ts = [x[0] for x in state]
    def ok(t):
        if not (t0 <= t <= t1): return False
        i = bisect.bisect_right(ts, t) - 1
        if i < 0: return False
        st_t, m, a = state[i]
        return m == 'AUTO' and a and (t - st_t) <= 2.0
    return ok

def q_at(v, p):
    v = sorted(v); k = (len(v) - 1) * p; f = math.floor(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

def replay(seq):
    """seq: list of (eligible_key or None, source_stamp). One trigger per eligible run.
    msg rule: two eligible messages; src rule: two distinct source stamps (|d|>1e-6)."""
    out = {}
    for rule in ('msg', 'src'):
        cnt = 0; last = None; key = None; fired = False; trig = 0
        for e, s in seq:
            if e is None: cnt = 0; last = None; key = None; fired = False; continue
            if e != key: key = e; cnt = 0; last = None; fired = False
            if rule == 'msg': cnt += 1
            elif last is None or abs(s - last) > 1e-6: cnt += 1; last = s
            if cnt >= 2 and not fired: trig += 1; fired = True
        out[rule] = trig
    return out

res = {'per_test': {}, 'class': collections.defaultdict(lambda: collections.Counter())}
all_ages = []; all_kappa = []; all_gaps = []; all_lat = []
for T in ('T1', 'T2', 'T3', 'T4'):
    R = pickle.load(open(f'inv/{T}.pkl', 'rb'))
    ok = armed_auto_selector(R['/mavros/state'], T)
    cam = [c for c in R['/camera/perception/status'] if ok(c['t'])]
    br = [b for b in R['/safety_bridge/status'] if ok(b['t'])]
    ages = [c['age'] for c in cam if c['age'] is not None]
    # distinct valid source frames (camera output carries one YOLO source stamp)
    frames = {}
    for c in cam:
        if c['src'] and c['valid']: frames.setdefault(round(c['src'], 3), c)
    S = sorted(frames)
    gaps = [b - a for a, b in zip(S, S[1:])]
    kappa = list(collections.Counter(round(b['csrc'], 3) for b in br if b['csrc']).values())
    lat = [b['stamp'] - b['csrc'] for b in br if b['csrc'] and b['stamp']]
    pt = dict(n_cam=len(cam), n_frames=len(S), n_bridge=len(br), age_med=st.median(ages),
              age_p99=q_at(ages, .99), age_max=max(ages), age_gt075=sum(a > 0.75 for a in ages),
              gap_med=st.median(gaps), gap_p95=q_at(gaps, .95),
              kappa_med=st.median(kappa), kappa_min=min(kappa), kappa_share2=sum(k >= 2 for k in kappa) / len(kappa),
              lat_med=st.median(lat), lat_p95=q_at(lat, .95), lat_max=max(lat))
    # class-level runs over distinct source frames and the corresponding replays on bridge snapshots
    for cls in CLASSES:
        crit = [any(o[0] == cls and str(o[1]).upper() == 'CRITICAL' for o in frames[s]['objs']) for s in S]
        runs = []; r = 0
        for x in crit + [False]:
            if x: r += 1
            elif r: runs.append(r); r = 0
        # transitions for q: consecutive distinct frames
        n_from = sum(1 for a in crit[:-1] if a); n_stay = sum(1 for a, b in zip(crit, crit[1:]) if a and b)
        critset = {s for s, x in zip(S, crit) if x}
        seq = [((cls if round(b['csrc'] or 0, 3) in critset else None), b['csrc'] or 0) for b in br]
        rp = replay(seq)
        C = res['class'][cls]
        C.update(frames=sum(crit), runs=len(runs), runs2=sum(L >= 2 for L in runs), single=sum(L == 1 for L in runs),
                 n_from=n_from, n_stay=n_stay, msg=rp['msg'], src=rp['src'], maxrun=0)
        C['maxrun'] = max(C['maxrun'], max(runs, default=0))
        pt[cls] = dict(runs=runs, msg=rp['msg'], src=rp['src'])
    # deployed-path replay: bridge snapshots that are actually person-CRITICAL in fusion
    seqp = [(('person' if (b['fstate'] == 'CRITICAL' and b['htype'] == 'person' and b['cfresh'] and (b['csrc'] or 0) > 0) else None), b['csrc'] or 0) for b in br]
    pt['deployed_person_replay'] = replay(seqp)
    res['per_test'][T] = pt
    all_ages += ages; all_kappa += kappa; all_gaps += gaps; all_lat += lat

print('Per-test summary')
for T, p in res['per_test'].items():
    print(T, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items() if k not in CLASSES})
print('\nPooled: outputs', len(all_ages), 'frames', len(all_gaps) + 4)
print(' kappa median', st.median(all_kappa), 'share>=2 %.4f' % (sum(k >= 2 for k in all_kappa) / len(all_kappa)), 'min', min(all_kappa), 'n', len(all_kappa))
print(' frame interval median %.3f p95 %.3f' % (st.median(all_gaps), q_at(all_gaps, .95)))
print(' source->bridge age median %.3f p95 %.3f p99 %.3f max %.3f' % (st.median(all_lat), q_at(all_lat, .95), q_at(all_lat, .99), max(all_lat)))
print('\nClass-level runs and replays (T1-T4)')
for cls, C in res['class'].items():
    if not C['frames']: continue
    q = C['n_stay'] / C['n_from'] if C['n_from'] else float('nan'); lo, hi = wilson(C['n_stay'], C['n_from'])
    print(f" {cls:9s} frames {C['frames']:3d} runs {C['runs']:2d} runs>=2 {C['runs2']:2d} single {C['single']:2d} maxrun {C['maxrun']:2d} "
          f"| msg {C['msg']:2d} src {C['src']:2d} | q={q:.2f} [{lo:.2f},{hi:.2f}] ({C['n_stay']}/{C['n_from']}) runs>=2 share {C['runs2']/C['runs']:.2f}")
print('\nFreshness curve, pooled armed-AUTO publications')
for tau in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
    k = sum(a > tau for a in all_ages)
    print(f" tau {tau:.2f}: exceed {k:4d}  ({100*k/len(all_ages):.3f}%)")
print('\nDeployed person replay per test:', {T: p['deployed_person_replay'] for T, p in res['per_test'].items()})

# ---- M4: exceedance episodes vs validity HOLD requests; M3 person-event decomposition ----
print('\nM4 exceedance episodes (exceedances <2 s apart merged), per tau')
ep_by_tau = {}
for tau in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
    eps = 0
    for T in ('T1', 'T2', 'T3', 'T4'):
        R = pickle.load(open(f'inv/{T}.pkl', 'rb')); ok = armed_auto_selector(R['/mavros/state'], T)
        ex = sorted(c['t'] for c in R['/camera/perception/status'] if ok(c['t']) and c['age'] is not None and c['age'] > tau)
        eps += sum(1 for i, t in enumerate(ex) if i == 0 or t - ex[i - 1] > 2.0)
    ep_by_tau[tau] = eps
print(' ', ep_by_tau)
print('\nValidity HOLD requests vs exceedances at 0.75 s')
for T in ('T1', 'T2'):
    R = pickle.load(open(f'inv/{T}.pkl', 'rb')); ok = armed_auto_selector(R['/mavros/state'], T)
    t0 = next(t for t, m, a in R['/mavros/state'] if m == 'AUTO')
    ex = [(round(c['t'] - t0, 3), c['age']) for c in R['/camera/perception/status'] if ok(c['t']) and c['age'] and c['age'] > 0.75]
    hr = [(round(b['t'] - t0, 3), b['action']) for b in R['/safety_bridge/status'] if b['action'].startswith('HOLD_REQUESTED')]
    print(' ', T, 'exceed', ex, 'requests', hr)
R = pickle.load(open('inv/T4.pkl', 'rb'))
hr = [b for b in R['/safety_bridge/status'] if 'SEMANTIC_CONFIRMED' in b['action']]
b = hr[0]; prev = [x for x in R['/safety_bridge/status'] if x['t'] < b['t'] and x['action'].startswith('SEMANTIC_CONFIRM_PERSON')]
s1 = prev[-1]['csrc'] if prev else None
print('\nPerson event: sigma1', s1, 'sigma2', b['csrc'], 'delta_sigma %.3f' % (b['csrc'] - s1), 'a2 at request %.3f' % (b['stamp'] - b['csrc']),
      'D=t_req-sigma1 %.3f' % (b['stamp'] - s1))
gaps_all = []
for T in ('T1', 'T2', 'T3', 'T4'):
    R = pickle.load(open(f'inv/{T}.pkl', 'rb')); ok = armed_auto_selector(R['/mavros/state'], T)
    S = sorted({round(c['src'], 3) for c in R['/camera/perception/status'] if ok(c['t']) and c['src'] and c['valid']})
    gaps_all += [round(y - x, 3) for x, y in zip(S, S[1:])]
print('frame-interval min %.3f, share <=0.2 s %.4f, top values' % (min(gaps_all), sum(g <= 0.2 for g in gaps_all) / len(gaps_all)), collections.Counter(gaps_all).most_common(6))

# ---- M3: pending-confirmation intervals and how they ended ----
print('\nM3 pending person-confirmation intervals (bridge snapshots with SEMANTIC_CONFIRM_PERSON_1/2)')
for T in ('T1', 'T2', 'T3', 'T4'):
    R = pickle.load(open(f'inv/{T}.pkl', 'rb')); B = R['/safety_bridge/status']
    t0 = next(t for t, m, a in R['/mavros/state'] if m == 'AUTO')
    i = 0
    while i < len(B):
        if B[i]['action'].startswith('SEMANTIC_CONFIRM_PERSON_1'):
            j = i
            while j + 1 < len(B) and B[j + 1]['action'].startswith('SEMANTIC_CONFIRM_PERSON_1'): j += 1
            nxt = B[j + 1] if j + 1 < len(B) else None
            print(f"  {T} t={B[i]['t']-t0:7.2f}s snapshots={j-i+1} sigma={B[i]['csrc']:.3f} pending(first->next) {nxt['t']-B[i]['t'] if nxt else float('nan'):.3f}s "
                  f"age_at_last={B[j]['stamp']-B[j]['csrc']:.3f}s ended_by={nxt['action'] if nxt else None} next_sigma_new={nxt and abs((nxt['csrc'] or 0)-B[i]['csrc'])>1e-6}")
            i = j + 1
        else: i += 1

# ---- Assumption A1: source stamps referenced by consecutive bridge snapshots never go backwards ----
print('\nA1 check: backward source-stamp steps in armed-AUTO bridge snapshots')
for T in ('T1', 'T2', 'T3', 'T4'):
    R = pickle.load(open(f'inv/{T}.pkl', 'rb')); ok = armed_auto_selector(R['/mavros/state'], T)
    s = [b['csrc'] for b in R['/safety_bridge/status'] if ok(b['t']) and b['csrc']]
    back = sum(1 for x, y in zip(s, s[1:]) if y < x - 1e-6)
    print(f'  {T}: snapshots with stamp {len(s)}, backward steps {back}')

# ---- replication factor vs rate ratio ----
gm = []; ks = []
for T in ('T1', 'T2', 'T3', 'T4'):
    R = pickle.load(open(f'inv/{T}.pkl', 'rb')); ok = armed_auto_selector(R['/mavros/state'], T)
    S = sorted({round(c['src'], 3) for c in R['/camera/perception/status'] if ok(c['t']) and c['src'] and c['valid']})
    gm += [y - x for x, y in zip(S, S[1:]) if y - x < 2.0]
    ks += list(collections.Counter(round(b['csrc'], 3) for b in R['/safety_bridge/status'] if ok(b['t']) and b['csrc']).values())
print('\nmean frame interval %.3f s (gaps<2s, n=%d); mean kappa %.3f; T_c/T_f = %.2f; kappa dist' % (st.mean(gm), len(gm), st.mean(ks), st.mean(gm) / 0.1),
      sorted(collections.Counter(ks).items()))

# ---- per-test summary for Supplementary Table S5 ----
print('\nPer-test S5: frames, mean kappa, share kappa>=2, mean interval, kappa_pred, age->bridge median/p95/max')
for T in ('T1', 'T2', 'T3', 'T4'):
    R = pickle.load(open(f'inv/{T}.pkl', 'rb')); ok = armed_auto_selector(R['/mavros/state'], T)
    br = [b for b in R['/safety_bridge/status'] if ok(b['t']) and b['csrc']]
    k = list(collections.Counter(round(b['csrc'], 3) for b in br).values())
    S = sorted({round(c['src'], 3) for c in R['/camera/perception/status'] if ok(c['t']) and c['src'] and c['valid']})
    g = [y - x for x, y in zip(S, S[1:]) if y - x < 2.0]
    lat = [b['stamp'] - b['csrc'] for b in br if b['stamp']]
    print(f"  {T}: snapshots {len(br)} stamps {len(k)} mean_kappa {st.mean(k):.3f} share2 {100*sum(x>=2 for x in k)/len(k):.2f}% "
          f"mean_gap {st.mean(g):.3f} pred {st.mean(g)/0.1:.2f} | age med {st.median(lat):.3f} p95 {q_at(lat,.95):.3f} max {max(lat):.3f}")
