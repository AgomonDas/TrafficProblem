import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
r = pd.read_csv("results.csv"); s = pd.read_csv("series.csv")
algos = ["Shortest-path", "Greedy-LB", "Proposed"]; col = {"Shortest-path": "#d62728", "Greedy-LB": "#ff9f1c", "Proposed": "#2a9d8f"}
inst = list(r.instance.unique()); x = np.arange(len(inst)); w = 0.26

def bars(ax, key, title, ylab, scale=1.0, hline=None):
    for i, a in enumerate(algos):
        v = [r[(r.instance == n) & (r.algo == a)][key].iloc[0] * scale for n in inst]
        ax.bar(x + (i - 1) * w, v, w, label=a, color=col[a])
    if hline: ax.axhline(hline, color="k", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(inst, fontsize=8); ax.set_title(title, fontsize=10); ax.set_ylabel(ylab)

fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
bars(ax[0,0], "usedPct", "UTILISATION: % of roads carrying traffic (higher = better)", "% of roads")
bars(ax[0,1], "maxUtil", "CONGESTION: worst road load / capacity (>100% = jam)", "% of capacity", 100, 100)
ax[0,1].set_yscale("log")
bars(ax[0,2], "totDist", "TOTAL DISTANCE covered by all vehicles", "distance units")
bars(ax[1,0], "overloaded", "Congestion: # (road, time-window) pairs over capacity", "count")
bars(ax[1,1], "varUtil", "Load balance: variance of road utilisation (lower = better)", "variance")
r["dpv"] = r.totDist / r.served
bars(ax[1,2], "dpv", "Detour cost: distance per served vehicle", "distance / vehicle")
for a_ in ax.flat: a_.grid(axis="y", alpha=.3)
ax[0,0].legend(); fig.tight_layout(); fig.savefig("fig1_comparison.png", dpi=150)

d = s[s.instance == "Rush-500"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for a in algos:
    q = d[d.algo == a]
    ax[0].plot(q.window, q.maxUtil * 100, label=a, color=col[a], marker="o")
    ax[1].plot(q.window, q.avgUtil * 100, label=a, color=col[a], marker="o")
    ax[2].plot(q.window, q.overloaded, label=a, color=col[a], marker="o")
ax[0].axhline(100, color="k", ls="--"); ax[0].set_yscale("log")
ax[0].set_title("Rush-500: worst road load per time window (% cap)"); ax[1].set_title("Rush-500: network utilisation per window (%)")
ax[2].set_title("Rush-500: # overloaded roads per window")
for a_ in ax: a_.set_xlabel("time window (10 ticks)"); a_.grid(alpha=.3)
ax[0].legend(); fig.tight_layout(); fig.savefig("fig2_rush_timeseries.png", dpi=150)

t = pd.read_csv("sweep.csv"); fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(t.alpha, t.varUtil, "o-", color="#2a9d8f"); ax.set_xlabel("alpha (balance weight)"); ax.set_ylabel("variance of utilisation", color="#2a9d8f")
ax2 = ax.twinx(); ax2.plot(t.alpha, t.distPerVeh, "s--", color="#d62728"); ax2.set_ylabel("distance per vehicle", color="#d62728")
ax.set_title("Trade-off: balance vs distance (Medium-200)"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("fig3_tradeoff.png", dpi=150)

c = pd.read_csv("scale.csv"); fig, ax = plt.subplots(figsize=(7, 4.5))
for k, a in [("sp_ms", "Shortest-path"), ("glb_ms", "Greedy-LB"), ("prop_ms", "Proposed")]: ax.loglog(c.V, c[k], "o-", label=a, color=col[a])
ax.set_xlabel("|V|"); ax.set_ylabel("ms (5 steps)"); ax.set_title("Scalability"); ax.grid(alpha=.3, which="both"); ax.legend()
fig.tight_layout(); fig.savefig("fig4_scalability.png", dpi=150)
