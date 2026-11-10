#!/usr/bin/env python3
"""Publication figures for the NICE 2027 paper.

Reads figure_data.mat (written by neuromorphic_cognitive_radar.m, see
export_figure_data) and writes vector PDFs sized for the IEEE two-column
layout (column 3.49 in, text 7.16 in) with Times-like fonts (STIX).

Usage: python3 make_figures.py figure_data.mat out_dir
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec, patches
import matplotlib.patheffects as pe
import numpy as np
import scipy.io as sio
from scipy import stats

D = sio.loadmat(sys.argv[1], squeeze_me=True)
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

COL_W, TXT_W = 3.49, 7.16
C_UNI, C_ALG, C_NEU, C_EXC = '#6e6d68', '#eb6834', '#2a78d6', '#1baf7a'
C_INK, C_INK2, C_GRID, C_BAND = '#1f1f1d', '#52514e', '#dcdbd7', '#ecebe7'
POL = [('Uniform', C_UNI, (0, (5, 1.6, 1, 1.6)), 1.1),
       ('Rule-based', C_ALG, (0, (4, 1.6)), 1.2),
       ('Neuromorphic', C_NEU, '-', 1.5)]

plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.frameon': False, 'legend.handlelength': 2.2,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out',
    'ytick.direction': 'out', 'axes.edgecolor': C_INK2, 'xtick.color': C_INK2,
    'ytick.color': C_INK2, 'axes.labelcolor': C_INK, 'text.color': C_INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
    'savefig.dpi': 600, 'lines.solid_capstyle': 'round'})

t = np.asarray(D['t'], float)
Ns = t.size


def ma(x, k=3):
    """Centred moving average that ignores NaN (edges use the available samples)."""
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    h = k // 2
    for i in range(x.size):
        w = x[max(0, i - h):i + h + 1]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = w.mean()
    return out


def grid(ax, axis='y'):
    ax.grid(True, axis=axis, color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def events(ax, y_text, cyc=True):
    on, off, cy = float(D['intf_on']), float(D['intf_off']), float(D['cyc_on'])
    ax.axvspan(on, off, color=C_BAND, zorder=0, lw=0)
    ax.text((on + cy) / 2, y_text, 'interf.', ha='center', va='top', fontsize=6.5, color=C_INK2)
    if cyc:
        ax.axvline(cy, color=C_INK2, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        ax.text(cy + 1, y_text, 'cyclist', ha='left', va='top', fontsize=6.5, color=C_INK2)


def panel_label(ax, s, x=-0.2, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')


def labels_below(fig, groups, labels, gap_in=0.04, size=8):
    """Sub-figure labels centred under each group of axes, on a common baseline below the axis labels."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    ext = []
    for axes in groups:
        pos = [a.get_position() for a in axes]
        y0 = min(inv.transform(a.get_tightbbox(rend))[0, 1] for a in axes)
        ext.append(((min(p.x0 for p in pos) + max(p.x1 for p in pos)) / 2, y0))
    y = min(e[1] for e in ext) - gap_in / fig.get_figheight()
    for (x, _), lab in zip(ext, labels):
        fig.text(x, y, lab, fontsize=size, ha='center', va='top')


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)


# ===================================================================== Fig. dynamics
def fig_dynamics():
    fig = plt.figure(figsize=(TXT_W, 1.85))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.33, left=0.06, right=0.995, bottom=0.2, top=0.9)

    # (a) attention agreement and cognitive reward
    ax = fig.add_subplot(gs[0])
    events(ax, 1.13)
    rho_m, rho_h = D['rho_m'], D['rho_h']
    ax.fill_between(t, rho_m - rho_h, rho_m + rho_h, color=C_INK2, alpha=0.15, lw=0)
    ax.plot(t, rho_m, color=C_INK, lw=1.3, label=r'$\rho_F$ (SNN vs rule-based field)')
    for p, (name, col, ls, lw) in list(enumerate(POL))[1:]:
        m, h = ma(D['rcog_m'][p]), ma(D['rcog_h'][p])
        ax.fill_between(t, m - h, m + h, color=col, alpha=0.18, lw=0)
        ax.plot(t, m, color=col, ls=ls, lw=lw, label=rf'$R^{{\mathrm{{cog}}}}$, {name.lower()}')
    ax.axhline(0, color=C_INK2, lw=0.5)
    ax.set(xlim=(1, Ns), ylim=(-0.45, 1.15), xlabel='CPI index', ylabel='Correlation / reward')
    ax.set_yticks(np.arange(-0.4, 1.01, 0.2))
    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.71), borderaxespad=0.2, frameon=True,
              facecolor='white', edgecolor='none', framealpha=0.9, borderpad=0.25, labelspacing=0.3)
    grid(ax)
    dyn_axes = [ax]

    # (b) detection probability
    ax = fig.add_subplot(gs[1])
    events(ax, 1.035)
    for p, (name, col, ls, lw) in enumerate(POL):
        m, h = ma(D['pd_m'][p]), ma(D['pd_h'][p])
        ax.fill_between(t, m - h, m + h, color=col, alpha=0.15, lw=0)
        ax.plot(t, m, color=col, ls=ls, lw=lw, label=name)
    ax.set(xlim=(1, Ns), ylim=(0.5, 1.04), xlabel='CPI index', ylabel=r'$P_D$ (moving targets)')
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_[::-1], l_[::-1], loc='lower right', borderaxespad=0.2, frameon=True, facecolor='white',
              edgecolor='none', framealpha=0.9, borderpad=0.25, labelspacing=0.3)
    grid(ax)
    dyn_axes.append(ax)

    # (c) OSPA
    ax = fig.add_subplot(gs[2])
    events(ax, 5.3)
    for p, (name, col, ls, lw) in enumerate(POL):
        m, h = ma(D['ospa_m'][p]), ma(D['ospa_h'][p])
        ax.fill_between(t, m - h, m + h, color=col, alpha=0.15, lw=0)
        ax.plot(t, m, color=col, ls=ls, lw=lw, label=name)
    ax.set(xlim=(1, Ns), ylim=(0, 5.35), xlabel='CPI index', ylabel='OSPA distance (m)')
    grid(ax)
    dyn_axes.append(ax)
    labels_below(fig, [[a] for a in dyn_axes], ['(a)', '(b)', '(c)'])
    save(fig, 'fig_dynamics')


# ===================================================================== Fig. kernel
def fig_kernel():
    K = np.asarray(D['kernel'], float)            # Doppler x CPI x run, / w_max
    v = np.asarray(D['v'], float)
    w0 = float(D['w_init'])
    sel = np.abs(v) <= 10.0
    dv = v[1] - v[0]
    dW = K.mean(axis=2) - w0
    fig = plt.figure(figsize=(COL_W, 2.45))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[3.0, 1.25], height_ratios=[0.07, 1],
                           wspace=0.14, hspace=0.42, left=0.14, right=0.985, bottom=0.17, top=0.9)
    ax = fig.add_subplot(gs[1, 0])
    vmin, vmax = float(dW[sel].min()), float(dW[sel].max())
    norm = colors.TwoSlopeNorm(vmin=min(vmin, -0.05), vcenter=0.0, vmax=max(vmax, 0.05))
    te = np.arange(0.5, Ns + 1.0, 1.0)
    ve = np.append(v[sel] - dv / 2, v[sel][-1] + dv / 2)
    pm = ax.pcolormesh(te, ve, dW[sel, :], cmap='RdBu_r', norm=norm, shading='flat', rasterized=True)
    names = [str(n) for n in np.atleast_1d(D['tgt_names'])]
    short = {'Vehicle': 'vehicle', 'Pedestrian': 'pedestrian', 'Cyclist': 'cyclist'}
    for k, n in enumerate(names):
        vr = np.asarray(D['tgt_vr'][k], float)
        ax.plot(t, vr, color=C_INK, lw=0.7, ls=(0, (2.5, 1.5)))
        first = np.flatnonzero(np.isfinite(vr))[0]
        ax.text(t[first] + 1 if k != 2 else t[first] - 1, vr[first] + 0.45, short.get(n, n),
                fontsize=6.5, color=C_INK, ha='left' if k != 2 else 'right', va='bottom',
                path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])
    ax.set(xlim=(0.5, Ns + 0.5), ylim=(-10, 10), xlabel='CPI index', ylabel='Radial velocity (m/s)')
    ax.set_xticks([1, 16, 32, 48, 64])
    cax = fig.add_subplot(gs[0, 0])
    cb = fig.colorbar(pm, cax=cax, orientation='horizontal')
    cb.set_ticks([-0.6, -0.3, 0, 0.15, 0.3], labels=['$-$0.6', '$-$0.3', '0', '0.15', '0.3'])
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
    cax.xaxis.set_ticks_position('top')
    cax.set_title(r'Weight change $\Delta W/w_{\max}$ since initialisation', fontsize=7, pad=12)

    # profile at the last CPI (mean and 95 % CI over runs)
    ax2 = fig.add_subplot(gs[1, 1], sharey=ax)
    nrun = K.shape[2]
    tc = stats.t.ppf(0.975, nrun - 1)
    for idx, col, lw, lab in [(0, '#9ec5f4', 1.0, 'CPI 1'), (Ns - 1, C_NEU, 1.4, f'CPI {Ns}')]:
        m = K[:, idx, :].mean(axis=1)
        h = tc * K[:, idx, :].std(axis=1, ddof=1) / np.sqrt(nrun)
        ax2.fill_betweenx(v, m - h, m + h, color=col, alpha=0.25, lw=0)
        ax2.plot(m, v, color=col, lw=lw, label=f'SNN, {lab}')
    sig = max(float(D['sigv']), float(D['vres_uniform']) / 2)
    vv = np.linspace(-10, 10, 400)
    ax2.plot(1 - np.exp(-vv ** 2 / (2 * sig ** 2)), vv, color=C_ALG, lw=1.1, ls=(0, (4, 1.6)),
             label='clutter-notch prior')
    ax2.axvline(w0, color=C_INK2, lw=0.6, ls=(0, (1, 1.5)))
    ax2.set(xlim=(0, 1.05), xlabel=r'$W/w_{\max}$')
    ax2.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.setp(ax2.get_yticklabels(), visible=False)
    grid(ax2, 'x')
    ax2.legend(loc='lower left', bbox_to_anchor=(-0.05, 1.0), borderaxespad=0, handlelength=1.6,
               labelspacing=0.25, fontsize=6.5)
    save(fig, 'fig_kernel')


# ===================================================================== Fig. scene + attention
def polar_mesh(r_edges, th_edges):
    R, TH = np.meshgrid(r_edges, np.deg2rad(th_edges), indexing='ij')
    return R * np.sin(TH), R * np.cos(TH)


def draw_fov(ax, Rmax, edges, lw=0.5, col=C_GRID, sectors=True):
    a = np.deg2rad(np.linspace(edges[0], edges[-1], 200))
    ax.plot(Rmax * np.sin(a), Rmax * np.cos(a), color=col, lw=lw)
    for e in (edges if sectors else [edges[0], edges[-1]]):
        ax.plot([0, Rmax * np.sin(np.deg2rad(e))], [0, Rmax * np.cos(np.deg2rad(e))], color=col, lw=lw)


def fig_scene_attention():
    Rmax = float(D['Rmax'])
    edges = np.asarray(D['sec_edges'], float)
    cpis = [2, 16, 64]
    fig = plt.figure(figsize=(TXT_W, 2.55))
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 2.05], wspace=0.1,
                              left=0.045, right=0.935, bottom=0.12, top=0.92)

    # ---- scenario
    ax = fig.add_subplot(outer[0])
    draw_fov(ax, Rmax, edges)
    for j in range(edges.size - 1):
        c = np.deg2rad((edges[j] + edges[j + 1]) / 2)
        ax.text((Rmax + 3) * np.sin(c), (Rmax + 3) * np.cos(c), str(j + 1), fontsize=6.5,
                color=C_INK2, ha='center', va='center')
    veg = np.asarray(D['veg_p'])
    ax.scatter(veg[:, 0], veg[:, 1], s=1.5, color='#5d8a5d', lw=0, label='vegetation')
    st = np.asarray(D['stat_p'])
    for cx, cy in [st[0:4].mean(axis=0), st[4:8].mean(axis=0)]:
        ax.add_patch(patches.Rectangle((cx - 1, cy - 2.4), 2, 4.8, fc='#8a8985', ec='none'))
    ax.plot(st[8, 0], st[8, 1], marker='^', ms=3.5, color='#8a8985', lw=0)
    ax.text(st[8, 0] + 2, st[8, 1], 'pole', fontsize=6.5, color=C_INK2, va='center')
    ip = np.asarray(D['intf_pos'], float)
    ax.plot([0, ip[0]], [0, ip[1]], color=C_ALG, lw=0.7, ls=(0, (2, 1.5)))
    ax.plot(ip[0], ip[1], marker='*', ms=7, color=C_ALG, mec='white', mew=0.3)
    ax.text(ip[0] - 1.5, ip[1] + 1.8, 'interferer', fontsize=6.5, color=C_ALG, ha='right')
    names = [str(n) for n in np.atleast_1d(D['tgt_names'])]
    lab_off = {0: (1.5, 1.2, 'left'), 1: (1.5, 1.0, 'left'), 2: (1.6, -1.5, 'left')}
    for k, n in enumerate(names):
        x, y = np.asarray(D['tgt_x'][k], float), np.asarray(D['tgt_y'][k], float)
        ok = np.isfinite(x)
        x, y = x[ok], y[ok]
        ax.plot(x, y, color=C_NEU, lw=1.3)
        ax.annotate('', xy=(x[-1], y[-1]), xytext=(x[-4], y[-4]),
                    arrowprops=dict(arrowstyle='-|>', color=C_NEU, lw=1.0, mutation_scale=6))
        ax.plot(x[0], y[0], marker='o', ms=3, mfc='white', mec=C_NEU, mew=0.9)
        dx, dy, ha = lab_off[k]
        ax.text(x[0] + dx, y[0] + dy, n, fontsize=6.5, ha=ha, va='center', color=C_INK)
    ax.text(28, 3.5, 'vegetation', fontsize=6.5, color='#3f6b3f', ha='center')
    ax.text(-27, 9.5, 'parked cars', fontsize=6.5, color=C_INK2, ha='center')
    ax.plot(0, 0, marker='^', ms=5, color=C_INK)
    ax.set_aspect('equal')
    ax.set(xlim=(-47, 47), ylim=(-1.5, 54), xlabel='$x$ (m)', ylabel='$y$ (m)')
    ax.set_xticks([-40, -20, 0, 20, 40]); ax.set_yticks([0, 25, 50])
    ax_scene = ax

    # ---- attention fields, shared colour scale (square-root compression)
    inner = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1], wspace=0.05, hspace=0.12)
    X, Y = polar_mesh(np.asarray(D['r_edges'], float), np.asarray(D['th_edges'], float))
    fields = {(r, c): np.asarray(D[f'{key}_{c}'], float) for r, key in enumerate(['Falg', 'Fneu']) for c in cpis}
    vmax = max(f.max() for f in fields.values())
    norm = colors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    cmap = plt.get_cmap('viridis')
    rows = ['Rule-based', 'Neuromorphic']
    att_axes = []
    for r in range(2):
        for ci, c in enumerate(cpis):
            a = fig.add_subplot(inner[r, ci])
            F = fields[(r, c)]
            pm = a.pcolormesh(X, Y, F, cmap=cmap, norm=norm, shading='flat', rasterized=True,
                              edgecolors='none')
            draw_fov(a, Rmax, edges, lw=0.4, col='#9a9a96', sectors=False)
            for k in range(len(names)):
                x, y = D['tgt_x'][k][c - 1], D['tgt_y'][k][c - 1]
                if np.isfinite(x):
                    a.plot(x, y, marker='o', ms=6.5, mfc='none', mec='white', mew=0.9)
                    a.plot(x, y, marker='o', ms=6.5, mfc='none', mec=C_ALG, mew=0.5)
            a.set_aspect('equal')
            a.set(xlim=(-45, 45), ylim=(-1.5, 52))
            a.set_xticks([-40, 0, 40]); a.set_yticks([0, 25, 50])
            if r == 0:
                a.set_title(f'CPI {c}', fontsize=8, pad=3)
                plt.setp(a.get_xticklabels(), visible=False)
            else:
                a.set_xlabel('$x$ (m)', labelpad=1)
            if ci > 0:
                plt.setp(a.get_yticklabels(), visible=False)
            else:
                a.set_ylabel(rows[r], fontsize=8, labelpad=3)
            for s in ['left', 'bottom']:
                a.spines[s].set_visible(False)
            a.tick_params(length=0, pad=1)
            att_axes.append(a)
    cax = fig.add_axes([0.945, 0.17, 0.011, 0.68])
    cb = fig.colorbar(pm, cax=cax)
    cb.set_label('Attention field', fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)

    labels_below(fig, [[ax_scene], att_axes], ['(a)', '(b)'])
    save(fig, 'fig_scene_attention')


# ===================================================================== Fig. sweeps
def fig_sweeps():
    e = np.asarray(D['eirp'], float)
    lab = ['off' if not np.isfinite(x) else f'{x:g}' for x in e]
    pol4 = [('Uniform', C_UNI, (0, (5, 1.6, 1, 1.6)), 's'), ('Rule-based', C_ALG, (0, (4, 1.6)), 'D'),
            ('Neuromorphic', C_NEU, '-', 'o'), ('Uniform + excision', C_EXC, (0, (1, 1.2)), '^')]
    fig = plt.figure(figsize=(TXT_W, 1.8))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.36, left=0.06, right=0.94, bottom=0.22, top=0.9)
    x = np.arange(e.size)
    sw_axes = []
    for q, (key, ylab, ylim) in enumerate([('sw_pdint', r'Vehicle $P_D$ (CPIs 20–44)', (0.45, 1.04)),
                                           ('sw_fa', 'False clusters per CPI', (0, 4.8))]):
        ax = fig.add_subplot(gs[q])
        for p, (name, col, ls, mk) in enumerate(pol4):
            m, h = D[f'{key}_m'][:, p], D[f'{key}_h'][:, p]
            xx = x + (p - 1.5) * 0.09
            ax.errorbar(xx, m, yerr=h, color=col, ls=ls, lw=1.2, marker=mk, ms=3.2, mfc=col, mec='white',
                        mew=0.4, capsize=0, elinewidth=0.7, label=name)
        ax.set_xticks(x, lab)
        ax.set(ylim=ylim, xlim=(-0.35, e.size - 0.65), xlabel='Interferer EIRP (dBm)', ylabel=ylab)
        grid(ax)
        sw_axes.append(ax)
        if q == 0:
            ax.legend(loc='lower left', borderaxespad=0.3)
    # (c) time steps per CPI
    ax = fig.add_subplot(gs[2])
    T = np.asarray(D['T'], float)
    ax.errorbar(T, D['lat_pd_m'], yerr=D['lat_pd_h'], color=C_NEU, lw=1.4, marker='o', ms=3.5,
                mec='white', mew=0.4, capsize=0, elinewidth=0.7, label=r'$P_D$')
    ax.set_xscale('log')
    ax.set_xticks(T, [f'{v:g}' for v in T])
    ax.minorticks_off()
    ax.set(ylim=(0.8, 1.0), xlabel='Network time steps $T$ per CPI', ylabel=r'$P_D$ (CPIs 16–64)')
    grid(ax)
    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.plot(T, np.asarray(D['lat_syn_m']) / 1e3, color=C_INK2, lw=1.0, ls=(0, (3, 1.5)), marker='s', ms=3,
             mfc='white', mec=C_INK2, mew=0.8, label='synaptic events')
    ax2.set_ylim(0, 2.0)
    ax2.set_ylabel(r'Events per CPI ($\times 10^3$)', color=C_INK2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, [r'$P_D$', 'synaptic events'], loc='lower right', borderaxespad=0.2)
    sw_axes.append(ax)
    labels_below(fig, [[a] for a in sw_axes], ['(a)', '(b)', '(c)'])
    save(fig, 'fig_sweeps')


fig_dynamics()
fig_kernel()
fig_scene_attention()
fig_sweeps()
print('figures written to', OUT)
