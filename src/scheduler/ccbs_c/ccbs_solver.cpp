#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr double EPS = 1e-9;
constexpr double INF = 1e100;

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        return (static_cast<std::size_t>(static_cast<unsigned int>(p.first)) << 32) ^
               static_cast<unsigned int>(p.second);
    }
};

long long state_key(int node, int interval_id) {
    return (static_cast<long long>(node) << 32) ^
           static_cast<unsigned int>(interval_id);
}

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct Interval {
    double first = 0.0;
    double second = 0.0;
};

struct Agent {
    int start = -1;
    int goal = -1;
    double radius = 0.0;
};

struct TimedState {
    int node = -1;
    double time = 0.0;
};

struct TimedMove {
    int u = -1;
    int v = -1;
    double t1 = 0.0;
    double t2 = 0.0;

    bool is_wait() const { return u == v; }
    double duration() const { return t2 - t1; }
};

struct Constraint {
    int agent = -1;
    double t1 = 0.0;
    double t2 = 0.0;
    int u = -1;
    int v = -1;

    bool is_vertex() const { return u == v; }
};

struct Conflict {
    int agent1 = -1;
    int agent2 = -1;
    TimedMove move1;
    TimedMove move2;
    double time = INF;
};

struct Path {
    int agent = -1;
    std::vector<TimedState> states;
    int expanded = 0;

    double cost() const {
        return states.empty() ? INF : states.back().time;
    }

    std::vector<TimedMove> moves(bool include_final_hold = true) const {
        std::vector<TimedMove> result;
        for (std::size_t i = 0; i + 1 < states.size(); ++i) {
            const auto& a = states[i];
            const auto& b = states[i + 1];
            if (b.time <= a.time + EPS) {
                continue;
            }
            result.push_back({a.node, b.node, a.time, b.time});
        }
        if (include_final_hold && !states.empty()) {
            const auto& last = states.back();
            result.push_back({last.node, last.node, last.time, INF});
        }
        return result;
    }
};

struct Graph {
    // motion_profile: 0 = linear, 1 = bang-bang.
    int motion_profile = 1;
    std::vector<int> external_ids;
    std::vector<Point> coords;
    std::vector<std::vector<std::pair<int, double>>> adj;
    std::vector<std::vector<std::pair<int, double>>> rev;
    std::unordered_map<int, int> id_to_index;

    void add_node(int external_id, Point p) {
        if (id_to_index.count(external_id)) {
            throw std::runtime_error("duplicate node id");
        }
        int idx = static_cast<int>(coords.size());
        id_to_index[external_id] = idx;
        external_ids.push_back(external_id);
        coords.push_back(p);
        adj.emplace_back();
        rev.emplace_back();
    }

    int index_of(int external_id) const {
        auto it = id_to_index.find(external_id);
        if (it == id_to_index.end()) {
            throw std::runtime_error("unknown node id");
        }
        return it->second;
    }

    void add_edge_external(int u_external, int v_external, double duration) {
        if (duration <= 0.0) {
            throw std::runtime_error("edge duration must be positive");
        }
        int u = index_of(u_external);
        int v = index_of(v_external);
        adj[u].push_back({v, duration});
        rev[v].push_back({u, duration});
    }
};

std::vector<Interval> merge_intervals(std::vector<Interval> intervals) {
    std::vector<Interval> cleaned;
    cleaned.reserve(intervals.size());
    for (auto interval : intervals) {
        interval.first = std::max(0.0, interval.first);
        if (interval.second > interval.first + EPS) {
            cleaned.push_back(interval);
        }
    }
    if (cleaned.empty()) {
        return {};
    }
    std::sort(cleaned.begin(), cleaned.end(), [](const Interval& a, const Interval& b) {
        if (std::abs(a.first - b.first) > EPS) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });

    std::vector<Interval> merged;
    merged.push_back(cleaned.front());
    for (std::size_t i = 1; i < cleaned.size(); ++i) {
        auto& last = merged.back();
        if (cleaned[i].first <= last.second + EPS) {
            last.second = std::max(last.second, cleaned[i].second);
        } else {
            merged.push_back(cleaned[i]);
        }
    }
    return merged;
}

std::vector<Interval> complement_intervals(const std::vector<Interval>& blocked) {
    auto blocks = merge_intervals(blocked);
    std::vector<Interval> safe;
    double cursor = 0.0;
    for (const auto& block : blocks) {
        if (cursor < block.first - EPS) {
            safe.push_back({cursor, block.first});
        }
        cursor = std::max(cursor, block.second);
        if (cursor >= INF / 2.0) {
            break;
        }
    }
    if (cursor < INF / 2.0) {
        safe.push_back({cursor, INF});
    }
    return safe;
}

std::optional<int> interval_containing(const std::vector<Interval>& intervals, double t) {
    for (int i = 0; i < static_cast<int>(intervals.size()); ++i) {
        if (intervals[i].first - EPS <= t && t <= intervals[i].second + EPS) {
            return i;
        }
    }
    return std::nullopt;
}

std::optional<double> earliest_departure(
    double current_time,
    Interval source_interval,
    Interval target_interval,
    double duration,
    const std::vector<Interval>& edge_blocks
) {
    double depart = std::max({current_time, source_interval.first, target_interval.first - duration});
    while (true) {
        if (depart > source_interval.second + EPS) {
            return std::nullopt;
        }
        double arrival = depart + duration;
        if (arrival < target_interval.first - EPS) {
            depart = target_interval.first - duration;
            continue;
        }
        if (arrival > target_interval.second + EPS) {
            return std::nullopt;
        }

        bool shifted = false;
        for (const auto& block : edge_blocks) {
            if (block.first - EPS <= depart && depart < block.second - EPS) {
                depart = block.second;
                shifted = true;
                break;
            }
        }
        if (!shifted) {
            return depart;
        }
    }
}

class RawSIPP {
public:
    explicit RawSIPP(const Graph& graph_) : graph(graph_) {}

    std::optional<Path> find_path(
        int agent_id,
        int start,
        int goal,
        const std::vector<Constraint>& constraints
    ) {
        std::unordered_map<int, std::vector<Interval>> vertex_blocks;
        std::unordered_map<std::pair<int, int>, std::vector<Interval>, PairHash> edge_blocks;

        for (const auto& con : constraints) {
            if (con.agent != agent_id || con.t2 <= con.t1 + EPS) {
                continue;
            }
            if (con.is_vertex()) {
                vertex_blocks[con.u].push_back({con.t1, con.t2});
            } else {
                edge_blocks[{con.u, con.v}].push_back({con.t1, con.t2});
            }
        }
        for (auto& item : vertex_blocks) {
            item.second = merge_intervals(item.second);
        }
        for (auto& item : edge_blocks) {
            item.second = merge_intervals(item.second);
        }

        std::unordered_map<int, std::vector<Interval>> safe_cache;
        auto safe_intervals = [&](int node) -> const std::vector<Interval>& {
            auto it = safe_cache.find(node);
            if (it != safe_cache.end()) {
                return it->second;
            }
            std::vector<Interval> blocked;
            auto block_it = vertex_blocks.find(node);
            if (block_it != vertex_blocks.end()) {
                blocked = block_it->second;
            }
            auto inserted = safe_cache.emplace(node, complement_intervals(blocked));
            return inserted.first->second;
        };

        const auto& start_intervals = safe_intervals(start);
        auto start_interval_id = interval_containing(start_intervals, 0.0);
        if (!start_interval_id.has_value()) {
            return std::nullopt;
        }

        const auto& heuristic = static_distances_to(goal);

        struct Record {
            int node = -1;
            int interval_id = -1;
            double time = 0.0;
            int parent = -1;
            double depart_time = -1.0;
        };
        struct OpenItem {
            double f = INF;
            double neg_time = 0.0;
            int counter = 0;
            int record = -1;
        };
        struct OpenGreater {
            bool operator()(const OpenItem& a, const OpenItem& b) const {
                if (std::abs(a.f - b.f) > EPS) {
                    return a.f > b.f;
                }
                if (std::abs(a.neg_time - b.neg_time) > EPS) {
                    return a.neg_time > b.neg_time;
                }
                return a.counter > b.counter;
            }
        };

        std::vector<Record> records;
        records.push_back({start, *start_interval_id, 0.0, -1, -1.0});
        std::priority_queue<OpenItem, std::vector<OpenItem>, OpenGreater> open;
        int counter = 0;
        open.push({heuristic[start], -0.0, counter, 0});

        std::unordered_map<long long, double> best_time;
        best_time[state_key(start, *start_interval_id)] = 0.0;
        std::unordered_set<long long> closed;
        int expanded = 0;

        while (!open.empty()) {
            auto item = open.top();
            open.pop();
            const auto current = records[item.record];
            long long key = state_key(current.node, current.interval_id);
            if (closed.count(key)) {
                continue;
            }
            auto best_it = best_time.find(key);
            if (best_it != best_time.end() && current.time > best_it->second + EPS) {
                continue;
            }
            closed.insert(key);
            expanded++;

            const auto& cur_intervals = safe_intervals(current.node);
            Interval cur_interval = cur_intervals[current.interval_id];
            if (current.node == goal && cur_interval.second >= INF / 2.0) {
                Path path = reconstruct_path(records, item.record);
                path.agent = agent_id;
                path.expanded = expanded;
                return path;
            }

            for (const auto& edge : graph.adj[current.node]) {
                int next = edge.first;
                double duration = edge.second;
                const auto& next_intervals = safe_intervals(next);
                std::vector<Interval> no_blocks;
                auto edge_it = edge_blocks.find({current.node, next});
                const auto& blocks = edge_it == edge_blocks.end() ? no_blocks : edge_it->second;

                for (int next_interval_id = 0;
                     next_interval_id < static_cast<int>(next_intervals.size());
                     ++next_interval_id) {
                    auto departure = earliest_departure(
                        current.time,
                        cur_interval,
                        next_intervals[next_interval_id],
                        duration,
                        blocks
                    );
                    if (!departure.has_value()) {
                        continue;
                    }
                    double arrival = *departure + duration;
                    long long succ_key = state_key(next, next_interval_id);
                    auto old = best_time.find(succ_key);
                    if (old != best_time.end() && arrival + EPS >= old->second) {
                        continue;
                    }
                    best_time[succ_key] = arrival;
                    records.push_back({next, next_interval_id, arrival, item.record, *departure});
                    counter++;
                    double h = heuristic[next];
                    open.push({arrival + h, -arrival, counter, static_cast<int>(records.size()) - 1});
                }
            }
        }

        return std::nullopt;
    }

private:
    const Graph& graph;
    std::unordered_map<int, std::vector<double>> heuristic_cache;

    const std::vector<double>& static_distances_to(int goal) {
        auto cached = heuristic_cache.find(goal);
        if (cached != heuristic_cache.end()) {
            return cached->second;
        }

        std::vector<double> dist(graph.coords.size(), INF);
        using HeapItem = std::pair<double, int>;
        std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;
        dist[goal] = 0.0;
        heap.push({0.0, goal});

        while (!heap.empty()) {
            auto [cost, node] = heap.top();
            heap.pop();
            if (cost > dist[node] + EPS) {
                continue;
            }
            for (const auto& edge : graph.rev[node]) {
                int pred = edge.first;
                double next_cost = cost + edge.second;
                if (next_cost + EPS < dist[pred]) {
                    dist[pred] = next_cost;
                    heap.push({next_cost, pred});
                }
            }
        }

        auto inserted = heuristic_cache.emplace(goal, std::move(dist));
        return inserted.first->second;
    }

    template <class Record>
    static Path reconstruct_path(const std::vector<Record>& records, int record_id) {
        std::vector<int> chain;
        int cur = record_id;
        while (cur >= 0) {
            chain.push_back(cur);
            cur = records[cur].parent;
        }
        std::reverse(chain.begin(), chain.end());

        Path path;
        if (chain.empty()) {
            return path;
        }
        path.states.push_back({records[chain[0]].node, records[chain[0]].time});
        for (std::size_t i = 1; i < chain.size(); ++i) {
            const auto& prev = records[chain[i - 1]];
            const auto& rec = records[chain[i]];
            double depart = rec.depart_time >= 0.0 ? rec.depart_time : prev.time;
            if (depart > prev.time + EPS) {
                path.states.push_back({prev.node, depart});
            }
            path.states.push_back({rec.node, rec.time});
        }
        return path;
    }
};

Point position_at(const Graph& graph, const TimedMove& move, double t);
double distance2_at(const Graph& graph, const TimedMove& a, const TimedMove& b, double t);

double move_fraction(const Graph& graph, const TimedMove& move, double t) {
    double alpha = std::clamp((t - move.t1) / (move.t2 - move.t1), 0.0, 1.0);
    if (graph.motion_profile == 1) {
        if (alpha <= 0.5) {
            return 2.0 * alpha * alpha;
        }
        double rem = 1.0 - alpha;
        return 1.0 - 2.0 * rem * rem;
    }
    return alpha;
}

Point position_at(const Graph& graph, const TimedMove& move, double t) {
    Point p0 = graph.coords[move.u];
    if (move.is_wait() || move.t2 <= move.t1 + EPS || move.t2 >= INF / 2.0) {
        return p0;
    }
    Point p1 = graph.coords[move.v];
    double alpha = move_fraction(graph, move, t);
    return {
        p0.x + alpha * (p1.x - p0.x),
        p0.y + alpha * (p1.y - p0.y),
    };
}

Point velocity(const Graph& graph, const TimedMove& move) {
    if (move.is_wait() || move.t2 <= move.t1 + EPS || move.t2 >= INF / 2.0) {
        return {0.0, 0.0};
    }
    Point p0 = graph.coords[move.u];
    Point p1 = graph.coords[move.v];
    double dt = move.t2 - move.t1;
    return {(p1.x - p0.x) / dt, (p1.y - p0.y) / dt};
}

std::optional<std::tuple<double, double, double>> linear_collision_interval(
    const Graph& graph,
    const TimedMove& move_a,
    const TimedMove& move_b,
    double radius_sum
) {
    double t0 = std::max(move_a.t1, move_b.t1);
    double t1 = std::min(move_a.t2, move_b.t2);
    if (t1 < t0 - EPS) {
        return std::nullopt;
    }

    Point pa = position_at(graph, move_a, t0);
    Point pb = position_at(graph, move_b, t0);
    Point va = velocity(graph, move_a);
    Point vb = velocity(graph, move_b);

    double px = pa.x - pb.x;
    double py = pa.y - pb.y;
    double vx = va.x - vb.x;
    double vy = va.y - vb.y;
    double r2 = radius_sum * radius_sum;

    double a = vx * vx + vy * vy;
    double b = 2.0 * (px * vx + py * vy);
    double c = px * px + py * py - r2;

    if (a <= EPS) {
        if (c <= EPS) {
            return std::make_tuple(t0, t1, t0);
        }
        return std::nullopt;
    }

    double closest_dt = -b / (2.0 * a);
    double overlap_end = t1 >= INF / 2.0 ? INF : t1 - t0;
    closest_dt = std::min(std::max(0.0, closest_dt), overlap_end);
    double closest_t = t0 + closest_dt;

    double disc = b * b - 4.0 * a * c;
    if (disc < -EPS) {
        return std::nullopt;
    }
    disc = std::max(0.0, disc);
    double root = std::sqrt(disc);
    double enter_dt = (-b - root) / (2.0 * a);
    double exit_dt = (-b + root) / (2.0 * a);
    double enter = std::max(enter_dt, 0.0);
    double exit = std::min(exit_dt, overlap_end);
    if (exit < enter - EPS) {
        return std::nullopt;
    }
    return std::make_tuple(t0 + enter, t0 + exit, closest_t);
}

std::tuple<double, double, double, double> move_bbox(const Graph& graph, const TimedMove& move) {
    Point p0 = graph.coords[move.u];
    Point p1 = graph.coords[move.v];
    return {
        std::min(p0.x, p1.x),
        std::max(p0.x, p1.x),
        std::min(p0.y, p1.y),
        std::max(p0.y, p1.y),
    };
}

double bbox_distance(
    const std::tuple<double, double, double, double>& a,
    const std::tuple<double, double, double, double>& b
) {
    double a_min_x, a_max_x, a_min_y, a_max_y;
    double b_min_x, b_max_x, b_min_y, b_max_y;
    std::tie(a_min_x, a_max_x, a_min_y, a_max_y) = a;
    std::tie(b_min_x, b_max_x, b_min_y, b_max_y) = b;
    double dx = std::max({b_min_x - a_max_x, a_min_x - b_max_x, 0.0});
    double dy = std::max({b_min_y - a_max_y, a_min_y - b_max_y, 0.0});
    return std::hypot(dx, dy);
}

double distance2_at(const Graph& graph, const TimedMove& a, const TimedMove& b, double t) {
    Point pa = position_at(graph, a, t);
    Point pb = position_at(graph, b, t);
    double dx = pa.x - pb.x;
    double dy = pa.y - pb.y;
    return dx * dx + dy * dy;
}

std::pair<double, double> golden_min_distance2(
    const Graph& graph,
    const TimedMove& move_a,
    const TimedMove& move_b,
    double lo,
    double hi
) {
    if (hi <= lo + EPS) {
        return {lo, distance2_at(graph, move_a, move_b, lo)};
    }
    constexpr double phi = 0.6180339887498948482;
    double x1 = hi - phi * (hi - lo);
    double x2 = lo + phi * (hi - lo);
    double f1 = distance2_at(graph, move_a, move_b, x1);
    double f2 = distance2_at(graph, move_a, move_b, x2);
    double tol = std::max(EPS, (hi - lo) * 1e-9);

    for (int i = 0; i < 64 && hi - lo > tol; ++i) {
        if (f1 <= f2) {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - phi * (hi - lo);
            f1 = distance2_at(graph, move_a, move_b, x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + phi * (hi - lo);
            f2 = distance2_at(graph, move_a, move_b, x2);
        }
    }

    std::vector<std::pair<double, double>> candidates = {
        {lo, distance2_at(graph, move_a, move_b, lo)},
        {hi, distance2_at(graph, move_a, move_b, hi)},
        {x1, f1},
        {x2, f2},
    };
    return *std::min_element(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
}

double collision_boundary(
    const Graph& graph,
    const TimedMove& move_a,
    const TimedMove& move_b,
    double safe_side,
    double colliding_side,
    double r2,
    bool want_left
) {
    if (distance2_at(graph, move_a, move_b, safe_side) <= r2 + EPS) {
        return safe_side;
    }
    double lo = std::min(safe_side, colliding_side);
    double hi = std::max(safe_side, colliding_side);
    for (int i = 0; i < 64; ++i) {
        double mid = 0.5 * (lo + hi);
        bool collides = distance2_at(graph, move_a, move_b, mid) <= r2 + EPS;
        if (want_left) {
            if (collides) {
                hi = mid;
            } else {
                lo = mid;
            }
        } else {
            if (collides) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
    return want_left ? hi : lo;
}

std::vector<double> collision_breakpoints(const TimedMove& move_a, const TimedMove& move_b, double t0, double t1) {
    std::vector<double> points = {t0, t1};
    for (const auto& move : {move_a, move_b}) {
        if (move.is_wait() || move.t2 >= INF / 2.0) {
            continue;
        }
        for (double t : {move.t1, 0.5 * (move.t1 + move.t2), move.t2}) {
            if (t0 < t && t < t1) {
                points.push_back(t);
            }
        }
    }
    double span = t1 - t0;
    for (int k = 1; k < 9; ++k) {
        points.push_back(t0 + span * static_cast<double>(k) / 9.0);
    }
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end(), [](double a, double b) {
        return std::abs(a - b) <= EPS;
    }), points.end());
    return points;
}

std::optional<std::tuple<double, double, double>> profiled_collision_interval(
    const Graph& graph,
    const TimedMove& move_a,
    const TimedMove& move_b,
    double radius_sum
) {
    double t0 = std::max(move_a.t1, move_b.t1);
    double t1 = std::min(move_a.t2, move_b.t2);
    if (t1 < t0 - EPS) {
        return std::nullopt;
    }

    if (bbox_distance(move_bbox(graph, move_a), move_bbox(graph, move_b)) > radius_sum + EPS) {
        return std::nullopt;
    }

    double r2 = radius_sum * radius_sum;
    if (t1 >= INF / 2.0) {
        double d2 = distance2_at(graph, move_a, move_b, t0);
        if (d2 <= r2 + EPS) {
            return std::make_tuple(t0, INF, t0);
        }
        return std::nullopt;
    }

    if (t1 <= t0 + EPS) {
        double d2 = distance2_at(graph, move_a, move_b, t0);
        if (d2 <= r2 + EPS) {
            return std::make_tuple(t0, t1, t0);
        }
        return std::nullopt;
    }

    auto points = collision_breakpoints(move_a, move_b, t0, t1);
    for (std::size_t i = 0; i + 1 < points.size(); ++i) {
        double lo = points[i];
        double hi = points[i + 1];
        if (hi < lo + EPS) {
            continue;
        }
        auto best = golden_min_distance2(graph, move_a, move_b, lo, hi);
        std::vector<std::pair<double, double>> candidates = {
            {lo, distance2_at(graph, move_a, move_b, lo)},
            {hi, distance2_at(graph, move_a, move_b, hi)},
            best,
        };
        auto best_it = std::min_element(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
        if (best_it->second > r2 + EPS) {
            continue;
        }
        double left = collision_boundary(graph, move_a, move_b, lo, best_it->first, r2, true);
        double right = collision_boundary(graph, move_a, move_b, best_it->first, hi, r2, false);
        return std::make_tuple(left, right, best_it->first);
    }

    return std::nullopt;
}

std::optional<std::tuple<double, double, double>> collision_interval(
    const Graph& graph,
    const TimedMove& move_a,
    const TimedMove& move_b,
    double radius_sum
) {
    if (graph.motion_profile == 1) {
        return profiled_collision_interval(graph, move_a, move_b, radius_sum);
    }
    return linear_collision_interval(graph, move_a, move_b, radius_sum);
}

struct Solution {
    bool found = false;
    std::vector<Path> paths;
    double flowtime = INF;
    double makespan = INF;
    int high_level_expanded = 0;
    int high_level_generated = 0;
    long long low_level_expanded = 0;
    double elapsed = 0.0;
};

struct CTNode {
    int id = 0;
    std::vector<Constraint> constraints;
    std::vector<Path> paths;
    double cost = 0.0;
    std::vector<Conflict> conflicts;
};

class ContinuousCBS {
public:
    ContinuousCBS(
        const Graph& graph_,
        std::vector<Agent> agents_,
        double precision_,
        double time_limit_,
        int max_high_level_nodes_,
        int high_level_order_
    )
        : graph(graph_),
          agents(std::move(agents_)),
          precision(precision_),
          time_limit(time_limit_),
          max_high_level_nodes(max_high_level_nodes_),
          high_level_order(high_level_order_),
          low_level(graph_) {}

    Solution find_solution() {
        auto started = std::chrono::steady_clock::now();
        Solution result;
        std::vector<Path> root_paths(agents.size());
        long long low_expanded = 0;

        for (int agent_id = 0; agent_id < static_cast<int>(agents.size()); ++agent_id) {
            auto path = low_level.find_path(
                agent_id,
                agents[agent_id].start,
                agents[agent_id].goal,
                {}
            );
            if (!path.has_value()) {
                result.elapsed = seconds_since(started);
                return result;
            }
            root_paths[agent_id] = *path;
            low_expanded += path->expanded;
        }

        CTNode root;
        root.id = 0;
        root.paths = root_paths;
        root.cost = sum_cost(root.paths);
        root.conflicts = get_all_conflicts(root.paths);

        std::vector<CTNode> nodes;
        nodes.push_back(std::move(root));

        std::priority_queue<HeapItem, std::vector<HeapItem>, HeapGreater> open;
        open.push(heap_item(nodes[0], 0));
        int generated = 1;
        int expanded = 0;
        std::unordered_set<std::string> seen;
        seen.insert("");

        while (!open.empty()) {
            if (seconds_since(started) > time_limit) {
                break;
            }
            if (expanded >= max_high_level_nodes) {
                break;
            }

            auto item = open.top();
            open.pop();
            const CTNode node = nodes[item.node_index];
            expanded++;

            auto conflicts = get_all_conflicts(node.paths);
            if (conflicts.empty()) {
                result.found = true;
                result.paths = node.paths;
                result.flowtime = sum_cost(node.paths);
                result.makespan = max_cost(node.paths);
                result.high_level_expanded = expanded;
                result.high_level_generated = generated;
                result.low_level_expanded = low_expanded;
                result.elapsed = seconds_since(started);
                return result;
            }

            const auto conflict_it = std::min_element(conflicts.begin(), conflicts.end(), [](const Conflict& a, const Conflict& b) {
                return a.time < b.time;
            });
            Conflict conflict = *conflict_it;

            for (int branch = 0; branch < 2; ++branch) {
                int replanned_agent = branch == 0 ? conflict.agent1 : conflict.agent2;
                int other_agent = branch == 0 ? conflict.agent2 : conflict.agent1;
                TimedMove own_move = branch == 0 ? conflict.move1 : conflict.move2;
                TimedMove other_move = branch == 0 ? conflict.move2 : conflict.move1;

                auto maybe_constraint = constraint_from_conflict(
                    replanned_agent,
                    other_agent,
                    own_move,
                    other_move
                );
                if (!maybe_constraint.has_value()) {
                    continue;
                }

                std::vector<Constraint> new_constraints = node.constraints;
                new_constraints.push_back(*maybe_constraint);
                std::string key = constraints_key(new_constraints);
                if (seen.count(key)) {
                    continue;
                }
                seen.insert(key);

                auto path = low_level.find_path(
                    replanned_agent,
                    agents[replanned_agent].start,
                    agents[replanned_agent].goal,
                    new_constraints
                );
                if (!path.has_value()) {
                    continue;
                }
                low_expanded += path->expanded;

                CTNode child;
                child.id = generated;
                child.constraints = std::move(new_constraints);
                child.paths = node.paths;
                child.paths[replanned_agent] = *path;
                child.cost = sum_cost(child.paths);
                child.conflicts = get_all_conflicts(child.paths);
                nodes.push_back(std::move(child));
                open.push(heap_item(nodes.back(), static_cast<int>(nodes.size()) - 1));
                generated++;
            }
        }

        result.found = false;
        result.high_level_expanded = expanded;
        result.high_level_generated = generated;
        result.low_level_expanded = low_expanded;
        result.elapsed = seconds_since(started);
        return result;
    }

private:
    const Graph& graph;
    std::vector<Agent> agents;
    double precision;
    double time_limit;
    int max_high_level_nodes;
    int high_level_order;
    RawSIPP low_level;

    static double seconds_since(std::chrono::steady_clock::time_point started) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }

    static double sum_cost(const std::vector<Path>& paths) {
        double total = 0.0;
        for (const auto& path : paths) {
            total += path.cost();
        }
        return total;
    }

    static double max_cost(const std::vector<Path>& paths) {
        double latest = 0.0;
        for (const auto& path : paths) {
            latest = std::max(latest, path.cost());
        }
        return latest;
    }

    struct HeapItem {
        double key1 = 0.0;
        double key2 = 0.0;
        int id = 0;
        int node_index = 0;
    };

    struct HeapGreater {
        bool operator()(const HeapItem& a, const HeapItem& b) const {
            if (std::abs(a.key1 - b.key1) > EPS) {
                return a.key1 > b.key1;
            }
            if (std::abs(a.key2 - b.key2) > EPS) {
                return a.key2 > b.key2;
            }
            return a.id > b.id;
        }
    };

    HeapItem heap_item(const CTNode& node, int node_index) const {
        if (high_level_order == 1) {
            return {static_cast<double>(node.conflicts.size()), node.cost, node.id, node_index};
        }
        return {node.cost, static_cast<double>(node.conflicts.size()), node.id, node_index};
    }

    std::vector<Conflict> get_all_conflicts(const std::vector<Path>& paths) const {
        std::vector<Conflict> conflicts;
        for (int i = 0; i < static_cast<int>(paths.size()); ++i) {
            for (int j = i + 1; j < static_cast<int>(paths.size()); ++j) {
                auto conflict = check_paths(i, paths[i], j, paths[j]);
                if (conflict.has_value()) {
                    conflicts.push_back(*conflict);
                }
            }
        }
        return conflicts;
    }

    std::optional<Conflict> check_paths(
        int agent1,
        const Path& path1,
        int agent2,
        const Path& path2
    ) const {
        std::optional<Conflict> best;
        double radius_sum = agents[agent1].radius + agents[agent2].radius;

        auto moves1 = path1.moves(true);
        auto moves2 = path2.moves(true);
        for (const auto& move1 : moves1) {
            for (const auto& move2 : moves2) {
                auto interval = collision_interval(graph, move1, move2, radius_sum);
                if (!interval.has_value()) {
                    continue;
                }
                double enter, exit, closest;
                std::tie(enter, exit, closest) = *interval;
                (void)exit;
                Conflict conflict;
                conflict.agent1 = agent1;
                conflict.agent2 = agent2;
                conflict.move1 = move1;
                conflict.move2 = move2;
                conflict.time = std::max(enter, closest);
                if (!best.has_value() || conflict.time < best->time) {
                    best = conflict;
                }
            }
        }
        return best;
    }

    std::optional<Constraint> constraint_from_conflict(
        int agent,
        int other_agent,
        const TimedMove& own_move,
        const TimedMove& other_move
    ) const {
        double radius_sum = agents[agent].radius + agents[other_agent].radius;
        if (own_move.is_wait()) {
            auto interval = collision_interval(graph, own_move, other_move, radius_sum);
            if (!interval.has_value()) {
                return std::nullopt;
            }
            double t1, t2, closest;
            std::tie(t1, t2, closest) = *interval;
            (void)closest;
            if (t2 <= t1 + EPS) {
                t2 = t1 + precision;
            }
            return Constraint{agent, t1, t2, own_move.u, own_move.u};
        }

        Interval interval = move_unsafe_start_interval(own_move, other_move, radius_sum);
        if (interval.second <= interval.first + EPS) {
            interval.second = interval.first + precision;
        }
        return Constraint{agent, interval.first, interval.second, own_move.u, own_move.v};
    }

    Interval move_unsafe_start_interval(
        const TimedMove& own_move,
        const TimedMove& other_move,
        double radius_sum
    ) const {
        if (other_move.t2 >= INF / 2.0) {
            return {own_move.t1, INF};
        }
        double duration = own_move.duration();
        if (!std::isfinite(duration) || duration <= 0.0) {
            return {own_move.t1, own_move.t1 + precision};
        }

        auto shifted = [&](double start_time) {
            return TimedMove{own_move.u, own_move.v, start_time, start_time + duration};
        };

        double low = own_move.t1;
        double high = std::max(low + precision, other_move.t2 + duration + precision);
        while (collision_interval(graph, shifted(high), other_move, radius_sum).has_value()) {
            high += std::max(duration, precision);
            if (high - own_move.t1 > 1e6) {
                return {own_move.t1, INF};
            }
        }

        for (int i = 0; i < 80; ++i) {
            if (high - low <= precision) {
                break;
            }
            double mid = 0.5 * (low + high);
            if (collision_interval(graph, shifted(mid), other_move, radius_sum).has_value()) {
                low = mid;
            } else {
                high = mid;
            }
        }
        return {own_move.t1, high};
    }

    static std::string constraints_key(std::vector<Constraint> constraints) {
        std::sort(constraints.begin(), constraints.end(), [](const Constraint& a, const Constraint& b) {
            if (a.agent != b.agent) return a.agent < b.agent;
            if (a.u != b.u) return a.u < b.u;
            if (a.v != b.v) return a.v < b.v;
            if (std::abs(a.t1 - b.t1) > EPS) return a.t1 < b.t1;
            return a.t2 < b.t2;
        });

        std::ostringstream out;
        out << std::setprecision(17);
        for (const auto& con : constraints) {
            out << con.agent << ':' << con.u << ':' << con.v << ':'
                << con.t1 << ':' << con.t2 << ';';
        }
        return out.str();
    }
};

Graph read_graph(int node_count, int edge_count, int motion_profile) {
    Graph graph;
    graph.motion_profile = motion_profile;
    for (int i = 0; i < node_count; ++i) {
        std::string label;
        int id;
        double x, y;
        std::cin >> label >> id >> x >> y;
        if (label != "node") {
            throw std::runtime_error("expected node line");
        }
        graph.add_node(id, {x, y});
    }
    for (int i = 0; i < edge_count; ++i) {
        std::string label;
        int u, v;
        double duration;
        std::cin >> label >> u >> v >> duration;
        if (label != "edge") {
            throw std::runtime_error("expected edge line");
        }
        graph.add_edge_external(u, v, duration);
    }
    return graph;
}

std::vector<Agent> read_agents(const Graph& graph, int agent_count) {
    std::vector<Agent> agents;
    agents.reserve(agent_count);
    for (int i = 0; i < agent_count; ++i) {
        std::string label;
        int start, goal;
        double radius;
        std::cin >> label >> start >> goal >> radius;
        if (label != "agent") {
            throw std::runtime_error("expected agent line");
        }
        agents.push_back({graph.index_of(start), graph.index_of(goal), radius});
    }
    return agents;
}

void write_solution(const Graph& graph, const Solution& solution) {
    std::cout << std::setprecision(17);
    std::cout << "FOUND " << (solution.found ? 1 : 0) << ' '
              << solution.flowtime << ' '
              << solution.makespan << ' '
              << solution.high_level_expanded << ' '
              << solution.high_level_generated << ' '
              << solution.low_level_expanded << ' '
              << solution.elapsed << '\n';
    if (!solution.found) {
        return;
    }
    std::cout << "PATHS " << solution.paths.size() << '\n';
    for (const auto& path : solution.paths) {
        std::cout << "PATH " << path.agent << ' ' << path.states.size() << '\n';
        for (const auto& state : path.states) {
            std::cout << "STATE " << graph.external_ids[state.node] << ' ' << state.time << '\n';
        }
    }
}

}  // namespace

int main() {
    try {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);

        std::string tag;
        std::cin >> tag;
        if (tag != "CCBS_C_V1") {
            throw std::runtime_error("expected CCBS_C_V1 header");
        }

        int node_count, edge_count, agent_count, max_high_level_nodes;
        int high_level_order, motion_profile;
        double precision, time_limit;
        std::cin >> node_count >> edge_count >> agent_count
                 >> precision >> time_limit >> max_high_level_nodes
                 >> high_level_order >> motion_profile;

        Graph graph = read_graph(node_count, edge_count, motion_profile);
        std::vector<Agent> agents = read_agents(graph, agent_count);

        ContinuousCBS solver(
            graph,
            std::move(agents),
            precision,
            time_limit,
            max_high_level_nodes,
            high_level_order
        );
        Solution solution = solver.find_solution();
        write_solution(graph, solution);
    } catch (const std::exception& exc) {
        std::cout << "ERROR " << exc.what() << '\n';
        return 1;
    }
    return 0;
}
