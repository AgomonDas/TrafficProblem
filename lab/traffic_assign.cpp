// Time-aware, capacity-, speed- and vehicle-type-aware traffic assignment.
// Build:  g++ -O2 -std=c++17 traffic_assign.cpp -o traffic
// Run:    ./traffic [demo|bench|sweep|scale|all]     (writes CSVs for plot_results.py)
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
using namespace std;

// ================================================================ model
// Road class = largest vehicle allowed. A truck road also takes bus/cng/cycle, etc.
enum VT { CYCLE = 0, CNG = 1, BUS = 2, TRUCK = 3 };
static const char* VN[4]  = {"cycle", "cng", "bus", "truck"};
static const int   PCU[4] = {1, 2, 4, 6};          // road space one vehicle occupies (passenger-car units)
static const int   BASECAP[4] = {8, 16, 28, 48};   // PCU that fit on a road of that class at the same instant

struct Edge { int u, v, dist, type, cap; };
struct Graph {
    int n = 0; vector<Edge> E; vector<vector<pair<int,int>>> adj;
    void init(int n_) { n = n_; adj.assign(n, {}); E.clear(); }
    int addEdge(int u, int v, int d, int t, int c) {
        E.push_back({u, v, d, t, c}); int id = (int)E.size() - 1;
        adj[u].push_back({v, id}); adj[v].push_back({u, id}); return id;
    }
};
struct Request { int t, n, s, d, type; };
struct Instance {
    string name; Graph g; int T = 0, delta = 10;          // delta = ticks per time step
    double speed[4] = {1.0, 3.0, 2.0, 1.5};               // distance units per tick, per vehicle type
    vector<vector<Request>> reqs;
};
static string V(int x) { return "v" + to_string(x + 1); }
static int travelTicks(const Instance* I, const Edge& e, int type) { return max(1, (int)ceil(e.dist / I->speed[type])); }

// ================================================================ task 1: generator
// rush=true: bursty demand (bell-shaped volume over time) with 70% of trips between a few
// hot-spot origins and destinations inside the heavy-road core -> a genuine stress test.
Instance generate(const string& name, int n, int T, double rLo, double rHi, int nmax,
                  unsigned seed, bool rush = false, int delta = 10) {
    mt19937 rng(seed);
    auto U  = [&]() { return uniform_real_distribution<double>(0, 1)(rng); };
    auto RI = [&](int a, int b) { return uniform_int_distribution<int>(a, b)(rng); };
    Instance I; I.name = name; I.T = T; I.delta = delta; Graph& g = I.g; g.init(n);
    vector<double> X(n), Y(n);
    for (int i = 0; i < n; i++) X[i] = U() * 100, Y[i] = U() * 100;
    double spacing = 100.0 / sqrt((double)n);
    auto eu   = [&](int a, int b) { return hypot(X[a] - X[b], Y[a] - Y[b]); };
    auto dist = [&](int a, int b) { return max(1, (int)lround(eu(a, b) / spacing * 4)); };
    auto cap  = [&](int t) { return BASECAP[t] * (U() < 0.3 ? 2 : 1); };       // 30% of roads have 2 lanes
    set<pair<int,int>> has;
    auto add = [&](int a, int b, int t) {
        if (a == b || !has.insert(minmax(a, b)).second) return;
        g.addEdge(a, b, dist(a, b), t, cap(t));
    };
    // Spanning tree => connected. Node i joins its nearest earlier node; class depends on i so that
    // nodes [0,.30n) are truck-connected, [0,.55n) bus-connected, [0,.80n) cng-connected, all cycle-connected.
    for (int i = 1; i < n; i++) {
        int bj = 0; double bd = 1e18;
        for (int j = 0; j < i; j++) { double d = eu(i, j); if (d < bd) bd = d, bj = j; }
        double f = (double)i / n; int t;
        if (f < 0.30) t = TRUCK;
        else if (f < 0.55) t = U() < 0.7 ? BUS : TRUCK;
        else if (f < 0.80) t = U() < 0.7 ? CNG : BUS;
        else t = U() < 0.6 ? CNG : CYCLE;
        add(i, bj, t);
    }
    // extra edges -> alternative routes (denser heavy-vehicle roads in the core)
    for (int rep = 0; rep < 3; rep++)
        for (int i = 0; i < n; i++) {
            vector<pair<double,int>> nb;
            for (int j = 0; j < n; j++) if (j != i) nb.push_back({eu(i, j), j});
            int k = min<int>(5, nb.size());
            partial_sort(nb.begin(), nb.begin() + k, nb.end());
            int j = nb[RI(0, k - 1)].second; double r = U();
            int ty = r < 0.15 ? TRUCK : r < 0.40 ? BUS : r < 0.75 ? CNG : CYCLE;
            int hi = max(i, j);
            if (hi < 0.30 * n && U() < 0.6) ty = TRUCK;
            else if (hi < 0.55 * n && ty < BUS && U() < 0.5) ty = BUS;
            add(i, j, ty);
        }
    // requests: endpoints always lie where a valid path exists for that vehicle type
    double frac[4] = {1.0, 0.8, 0.55, 0.30};
    vector<int> hotS, hotD; int core = max(4, (int)(0.30 * n));
    for (int i = 0; i < 3; i++) hotS.push_back(RI(0, core - 1));
    for (int i = 0; i < 3; i++) hotD.push_back(RI(0, core - 1));
    I.reqs.assign(T, {});
    for (int t = 0; t < T; t++) {
        double prof = rush ? 0.3 + 1.7 * exp(-pow((t - T / 2.0) / (T / 4.0), 2)) : 1.0;
        int R = (int)(RI(max(1, (int)(rLo * n)), max(1, (int)(rHi * n))) * prof);
        for (int k = 0; k < max(1, R); k++) {
            double r = U(); int ty = r < 0.15 ? CYCLE : r < 0.60 ? CNG : r < 0.87 ? BUS : TRUCK;
            int lim = max(2, (int)(n * frac[ty])), s = RI(0, lim - 1), d = RI(0, lim - 1);
            if (rush && U() < 0.7) { s = hotS[RI(0, 2)]; d = hotD[RI(0, 2)]; }
            while (d == s) d = RI(0, lim - 1);
            I.reqs[t].push_back({t, RI(1, nmax), s, d, ty});
        }
    }
    return I;
}

// ================================================================ time-indexed occupancy
// a[e][tick] = PCU on edge e at that tick. This is how "cng_2 may use e4 because cng_1 already left it" works.
struct Occ {
    vector<vector<int>> a; vector<array<int,4>> log; bool keepLog = false;
    Occ(int E = 0) : a(E) {}
    int peak(int e, int s, int t) const {
        const auto& v = a[e]; int m = 0, hi = min<int>(t, v.size());
        for (int i = s; i < hi; i++) m = max(m, v[i]);
        return m;
    }
    void add(int e, int s, int t, int w) {
        auto& v = a[e]; if ((int)v.size() < t) v.resize(t, 0);
        for (int i = s; i < t; i++) v[i] += w;
        if (keepLog) log.push_back({e, s, t, w});
    }
    int horizon() const { size_t h = 0; for (auto& v : a) h = max(h, v.size()); return (int)h; }
};

// ================================================================ task 2: routing core
enum Mode { SHORTEST, GREEDY_LB, PROPOSED };
static const char* MN[3] = {"Shortest-path", "Greedy-LB", "Proposed"};

struct Router {
    const Instance& I; const Graph& g; Occ& occ;
    vector<double> c1, c2; vector<int> dst, tim, pe, pv, stamp; int cur = 0;
    Router(const Instance& i, Occ& o) : I(i), g(i.g), occ(o), c1(g.n), c2(g.n), dst(g.n), tim(g.n), pe(g.n), pv(g.n), stamp(g.n, 0) {}
    // One vehicle of class `type` leaving s at tick t0. Arrival tick at node u = t0 + tim[u].
    // Each edge is checked against the occupancy *during the ticks the vehicle would be on it*.
    bool run(int s, int d, int type, int t0, Mode m, double alpha, int maxDist,
             vector<int>& path, int& pd, int& pt) {
        ++cur; typedef tuple<double,double,int> Q;
        priority_queue<Q, vector<Q>, greater<Q>> pq; int w = PCU[type];
        stamp[s] = cur; c1[s] = c2[s] = 0; dst[s] = tim[s] = 0; pe[s] = pv[s] = -1; pq.push({0, 0, s});
        while (!pq.empty()) {
            auto [a, b, u] = pq.top(); pq.pop();
            if (stamp[u] != cur || a > c1[u] || (a == c1[u] && b > c2[u])) continue;
            if (u == d) break;
            int tm = t0 + tim[u];
            for (auto [v, e] : g.adj[u]) {
                const Edge& ed = g.E[e];
                if (ed.type < type) continue;                              // road-type restriction
                int nd = dst[u] + ed.dist; if (nd > maxDist) continue;     // bounded detour
                int te = travelTicks(&I, ed, type);                        // speed-dependent crossing time
                double na, nb;
                if (m == SHORTEST) { na = a + ed.dist; nb = 0; }
                else {
                    int pk = occ.peak(e, tm, tm + te) + w;
                    if (pk > ed.cap) continue;                             // capacity (time-aware)
                    double ut = (double)pk / ed.cap;
                    if (m == PROPOSED) { na = a + ed.dist * (1 + alpha * ut * ut); nb = nd; }
                    else { na = max(a, ut); nb = nd; }                     // greedy: bottleneck first, then distance
                }
                if (stamp[v] != cur || na < c1[v] || (na == c1[v] && nb < c2[v])) {
                    stamp[v] = cur; c1[v] = na; c2[v] = nb; dst[v] = nd; tim[v] = tim[u] + te;
                    pe[v] = e; pv[v] = u; pq.push({na, nb, v});
                }
            }
        }
        if (stamp[d] != cur) return false;
        path.clear(); for (int x = d; x != s; x = pv[x]) path.push_back(pe[x]);
        reverse(path.begin(), path.end()); pd = dst[d]; pt = tim[d]; return true;
    }
};

struct Params { double alpha = 4.0; double stretch = 2.0; int maxDelay = 30; };
struct Assignment { vector<int> edges; int delay, count, dist, time; };
struct ReqRes { Request r; vector<Assignment> asg; int rejected = 0; bool noPath = false; };
struct Run { vector<vector<ReqRes>> steps; Occ occ; double ms = 0; };

static void commit(const Instance& I, Occ& occ, const vector<int>& path, int t0, int type, int cnt = 1) {
    int tm = t0;
    for (int e : path) { int te = travelTicks(&I, I.g.E[e], type); occ.add(e, tm, tm + te, PCU[type] * cnt); tm += te; }
}

Run solve(const Instance& I, Mode m, const Params& P) {
    Run R; R.occ = Occ(I.g.E.size()); R.occ.keepLog = I.g.E.size() < 10;
    Router rt(I, R.occ); R.steps.assign(I.T, {});
    auto st = chrono::steady_clock::now();
    for (int t = 0; t < I.T; t++) {
        auto rq = I.reqs[t];   // most constrained first: heavy vehicles, then biggest requests
        sort(rq.begin(), rq.end(), [](const Request& a, const Request& b) {
            return a.type != b.type ? a.type > b.type : a.n > b.n; });
        for (auto& r : rq) {
            ReqRes rr; rr.r = r; vector<int> path; int pd, pt, t0 = t * I.delta;
            if (!rt.run(r.s, r.d, r.type, t0, SHORTEST, 0, INT_MAX, path, pd, pt)) {   // free-flow shortest
                rr.rejected = r.n; rr.noPath = true; R.steps[t].push_back(rr); continue;
            }
            int sp = pd; map<pair<vector<int>,int>, int> cnt;
            if (m == SHORTEST) {                            // everybody on the one shortest path, capacity ignored
                commit(I, R.occ, path, t0, r.type, r.n); cnt[{path, 0}] = r.n;
            } else {
                int maxD = (m == PROPOSED) ? (int)ceil(P.stretch * sp) : INT_MAX;
                for (int v = 0; v < r.n; v++) {             // vehicle by vehicle, loads updated after each one
                    bool ok = false;
                    for (int delay = 0; delay <= P.maxDelay && !ok; delay += 2)
                        if (rt.run(r.s, r.d, r.type, t0 + delay, m, P.alpha, maxD, path, pd, pt)) {
                            commit(I, R.occ, path, t0 + delay, r.type); cnt[{path, delay}]++; ok = true;
                        }
                    if (!ok) rr.rejected++;
                }
            }
            for (auto& [k, c] : cnt) {
                int d = 0, tt = 0;
                for (int e : k.first) { d += I.g.E[e].dist; tt += travelTicks(&I, I.g.E[e], r.type); }
                rr.asg.push_back({k.first, k.second, c, d, tt});
            }
            R.steps[t].push_back(rr);
        }
    }
    R.ms = chrono::duration<double, milli>(chrono::steady_clock::now() - st).count();
    return R;
}

// ================================================================ task 4: metrics
struct Metrics {
    double maxLoad = 0, avgLoad = 0, varLoad = 0;
    double avgUtil = 0;      // mean over (edge,window) of min(1, load/cap): capacity actually put to use
    double usedPct = 0;      // % of roads that carry traffic at some point
    double maxUtil = 0;      // worst load/cap (>100% = jam)
    double varUtil = 0;      // load-balance: variance of utilisation across roads
    double excess = 0;       // sum over (edge,window) of max(0, load-cap)/cap : "how much jam"
    long overloaded = 0;     // number of (edge,window) pairs above capacity
    double totalDist = 0, totalTime = 0, ms = 0; long served = 0, rejected = 0;
};
Metrics evaluate(const Instance& I, const Run& R, int H, vector<array<double,4>>* series = nullptr) {
    Metrics M; int E = I.g.E.size(); vector<char> used(E, 0);
    for (int w = 0; w < H; w++) {
        double sL = 0, sU = 0, qL = 0, qU = 0, mx = 0, mu = 0, sC = 0, ex = 0; long ov = 0;
        for (int e = 0; e < E; e++) {
            double L = R.occ.peak(e, w * I.delta, (w + 1) * I.delta), c = I.g.E[e].cap, u = L / c;
            sL += L; sU += u; qL += L * L; qU += u * u; sC += min(1.0, u); mx = max(mx, L); mu = max(mu, u);
            if (L > c) { ov++; ex += (L - c) / c; }
            if (L > 0) used[e] = 1;
        }
        double mL = sL / E, mU = sU / E;
        M.maxLoad = max(M.maxLoad, mx); M.maxUtil = max(M.maxUtil, mu);
        M.avgLoad += mL / H; M.avgUtil += sC / E / H; M.varLoad += (qL / E - mL * mL) / H; M.varUtil += (qU / E - mU * mU) / H;
        M.overloaded += ov; M.excess += ex;
        if (series) series->push_back({sC / E, mu, (double)ov, 0});
    }
    for (int e = 0; e < E; e++) M.usedPct += used[e]; M.usedPct = 100.0 * M.usedPct / E;
    for (int t = 0; t < I.T; t++)
        for (auto& rr : R.steps[t]) {
            M.rejected += rr.rejected; double d = 0;
            for (auto& a : rr.asg) { M.served += a.count; M.totalDist += (double)a.count * a.dist; M.totalTime += (double)a.count * a.time; d += (double)a.count * a.dist; }
            if (series) (*series)[t][3] += d;
        }
    M.ms = R.ms; return M;
}
// Run all three algorithms on one instance over a common evaluation horizon
struct Trio { Run r[3]; Metrics m[3]; int H; };
Trio runAll(const Instance& I, const Params& P, bool wantSeries = false, ostream* ser = nullptr) {
    Trio X; X.H = I.T;
    for (int k = 0; k < 3; k++) { X.r[k] = solve(I, (Mode)k, P); X.H = max(X.H, (X.r[k].occ.horizon() + I.delta - 1) / I.delta); }
    for (int k = 0; k < 3; k++) {
        vector<array<double,4>> s; X.m[k] = evaluate(I, X.r[k], X.H, wantSeries ? &s : nullptr);
        if (ser) for (int w = 0; w < (int)s.size(); w++)
            *ser << I.name << "," << MN[k] << "," << w << "," << s[w][0] << "," << s[w][1] << "," << s[w][2] << "," << s[w][3] << "\n";
    }
    return X;
}
static void header() {
    printf("%-11s %-13s %7s %8s %8s %8s %7s %7s %8s %9s %9s %9s %8s %9s\n", "Instance", "Algo", "MaxLoad", "AvgLoad", "VarLoad",
           "AvgUtil", "Used", "MaxUtil", "VarUtil", "Overload", "TotDist", "TotTime", "Rejected", "Time(ms)");
}
static void row(const string& n, const char* a, const Metrics& M) {
    printf("%-11s %-13.13s %7.0f %8.2f %8.2f %7.1f%% %6.1f%% %6.1f%% %8.4f %9ld %9.0f %9.0f %8ld %9.1f\n", n.c_str(), a, M.maxLoad, M.avgLoad,
           M.varLoad, 100 * M.avgUtil, M.usedPct, 100 * M.maxUtil, M.varUtil, M.overloaded, M.totalDist, M.totalTime, M.rejected, M.ms);
}

// ================================================================ task 3: reports
static string pathStr(const Instance& I, int s, const vector<int>& p) {
    string o = V(s); int x = s;
    for (int e : p) { x = I.g.E[e].u == x ? I.g.E[e].v : I.g.E[e].u; o += "->" + V(x); }
    return o;
}
void printReport(ostream& os, const Instance& I, const Run& R, int topEdges, int maxSteps) {
    for (int t = 0; t < min(I.T, maxSteps); t++) {
        os << "=== time step " << t << " (ticks " << t * I.delta << ".." << (t + 1) * I.delta - 1 << ") ===\n";
        for (auto& rr : R.steps[t]) {
            auto& r = rr.r;
            os << "request (n=" << r.n << ", " << V(r.s) << ", " << V(r.d) << ", " << VN[r.type] << ")\n";
            if (rr.noPath) { os << "   no valid path for this vehicle type -> " << rr.rejected << " rejected\n"; continue; }
            for (auto& a : rr.asg)
                os << "   " << a.count << " x " << pathStr(I, r.s, a.edges) << "  dist=" << a.dist << " time=" << a.time
                   << (a.delay ? "  (departs +" + to_string(a.delay) + " ticks)" : "") << "\n";
            if (rr.rejected) os << "   " << rr.rejected << " vehicle(s) rejected (capacity)\n";
        }
        vector<tuple<double,int,int>> L;
        for (int e = 0; e < (int)I.g.E.size(); e++) {
            int l = R.occ.peak(e, t * I.delta, (t + 1) * I.delta); if (l > 0) L.push_back({-(double)l / I.g.E[e].cap, l, e});
        }
        sort(L.begin(), L.end()); os << "road loads (peak PCU in window, top " << topEdges << " by utilisation):\n";
        for (int i = 0; i < (int)L.size() && i < topEdges; i++) {
            auto [u, l, e] = L[i]; const Edge& ed = I.g.E[e];
            os << "   e" << e + 1 << " (" << V(ed.u) << "," << V(ed.v) << ") " << VN[ed.type] << "-road dist=" << ed.dist
               << "  load " << l << "/" << ed.cap << " = " << fixed << setprecision(0) << -100 * u << "%\n";
        }
    }
}

// ================================================================ drivers
void demo() {
    puts("################ DEMO: the 4-vertex example from the problem statement ################");
    Instance I; I.name = "demo"; I.T = 2; I.delta = 10; I.g.init(4);
    for (int i = 0; i < 4; i++) I.speed[i] = 1.0;                // 1 distance unit per tick => time = distance
    I.g.addEdge(0, 1, 3, CNG, 4);     // e1 (v1,v2) (3,cng)
    I.g.addEdge(0, 2, 3, CNG, 2);     // e2 (v1,v3) (3,cng)   room for ONE cng at a time
    I.g.addEdge(1, 2, 5, BUS, 8);     // e3 (v2,v3) (5,bus)
    I.g.addEdge(2, 3, 5, TRUCK, 2);   // e4 (v3,v4) (5,truck) room for ONE cng at a time -> must time-share
    I.reqs = {{{0, 2, 0, 3, CNG}}, {{1, 1, 1, 3, CNG}, {1, 1, 0, 3, TRUCK}}};
    Run R = solve(I, PROPOSED, Params());
    printReport(cout, I, R, 10, 2);
    puts("edge occupancy intervals [start,end) in ticks:");
    for (auto& l : R.occ.log) printf("   e%d: [%d,%d) +%d PCU\n", l[0] + 1, l[1], l[2], l[3]);
}
vector<Instance> suite() {
    vector<Instance> v;
    v.push_back(generate("Small-30", 30, 6, 0.25, 0.50, 6, 1));
    v.push_back(generate("Medium-200", 200, 10, 0.05, 0.10, 8, 2));
    v.push_back(generate("Large-800", 800, 10, 0.03, 0.06, 8, 3));
    v.push_back(generate("Rush-500", 500, 12, 0.05, 0.09, 10, 4, true));   // the complex scenario
    return v;
}
void bench() {
    puts("\n################ BENCHMARK (same instances for all algorithms) ################");
    header(); ofstream csv("results.csv"), ser("series.csv");
    csv << "instance,V,E,vehicles,algo,maxLoad,avgLoad,varLoad,avgUtil,usedPct,maxUtil,varUtil,overloaded,excess,totDist,totTime,served,rejected,ms\n";
    ser << "instance,algo,window,avgUtil,maxUtil,overloaded,dist\n";
    for (auto& I : suite()) {
        long veh = 0; for (auto& s : I.reqs) for (auto& r : s) veh += r.n;
        printf("[%s: |V|=%d |E|=%zu steps=%d vehicles=%ld]\n", I.name.c_str(), I.g.n, I.g.E.size(), I.T, veh);
        Trio X = runAll(I, Params(), true, &ser);
        for (int k = 0; k < 3; k++) {
            const Metrics& M = X.m[k]; row(I.name, MN[k], M);
            csv << I.name << "," << I.g.n << "," << I.g.E.size() << "," << veh << "," << MN[k] << "," << M.maxLoad << "," << M.avgLoad << ","
                << M.varLoad << "," << M.avgUtil << "," << M.usedPct << "," << M.maxUtil << "," << M.varUtil << "," << M.overloaded << ","
                << M.excess << "," << M.totalDist << "," << M.totalTime << "," << M.served << "," << M.rejected << "," << M.ms << "\n";
        }
        if (I.name == "Small-30" || I.name == "Rush-500")
            for (int k = 0; k < 3; k++) { ofstream f("report_" + I.name + "_" + MN[k] + ".txt"); printReport(f, I, X.r[k], I.name == "Small-30" ? 15 : 8, I.name == "Small-30" ? I.T : 3); }
    }
}
void sweep() {
    puts("\n################ TRADE-OFF: sweep of alpha (Medium-200) ################");
    Instance I = generate("Medium-200", 200, 10, 0.05, 0.10, 8, 2); ofstream csv("sweep.csv");
    csv << "alpha,maxUtil,avgUtil,varUtil,totDist,distPerVeh,rejected,ms\n";
    printf("%-8s %8s %8s %8s %9s %9s %9s %8s\n", "alpha", "MaxUtil", "AvgUtil", "VarUtil", "TotDist", "Dist/veh", "Rejected", "ms");
    for (double a : {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0}) {
        Params P; P.alpha = a; Run R = solve(I, PROPOSED, P);
        Metrics M = evaluate(I, R, max(I.T, (R.occ.horizon() + I.delta - 1) / I.delta));
        printf("%-8.1f %7.1f%% %7.1f%% %8.4f %9.0f %9.2f %9ld %8.1f\n", a, 100 * M.maxUtil, 100 * M.avgUtil, M.varUtil, M.totalDist, M.totalDist / M.served, M.rejected, M.ms);
        csv << a << "," << M.maxUtil << "," << M.avgUtil << "," << M.varUtil << "," << M.totalDist << "," << M.totalDist / M.served << "," << M.rejected << "," << M.ms << "\n";
    }
}
void scale() {
    puts("\n################ SCALABILITY (T=5 steps) ################");
    printf("%-6s %-7s %-9s %9s %9s %9s\n", "|V|", "|E|", "vehicles", "SP(ms)", "GLB(ms)", "Prop(ms)");
    ofstream csv("scale.csv"); csv << "V,E,vehicles,sp_ms,glb_ms,prop_ms\n"; double pt = 0; int pn = 0;
    for (int n : {50, 100, 200, 400, 800, 1600}) {
        Instance I = generate("s", n, 5, 0.05, 0.10, 8, 7); long veh = 0;
        for (auto& s : I.reqs) for (auto& r : s) veh += r.n;
        double ms[3]; for (int m = 0; m < 3; m++) ms[m] = solve(I, (Mode)m, Params()).ms;
        printf("%-6d %-7zu %-9ld %9.1f %9.1f %9.1f", n, I.g.E.size(), veh, ms[0], ms[1], ms[2]);
        if (pn) printf("   exponent~%.2f", log(ms[2] / pt) / log((double)n / pn));
        puts(""); pt = ms[2]; pn = n;
        csv << n << "," << I.g.E.size() << "," << veh << "," << ms[0] << "," << ms[1] << "," << ms[2] << "\n";
    }
}
int main(int argc, char** argv) {
    string m = argc > 1 ? argv[1] : "all";
    if (m == "demo" || m == "all") demo();
    if (m == "bench" || m == "all") bench();
    if (m == "sweep" || m == "all") sweep();
    if (m == "scale" || m == "all") scale();
}
