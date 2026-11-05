// USI-Chu engine mode of chushogi-az (USI-CHU.md §3-5).
//
// The main thread reads commands and `go` searches on a worker thread, so
// `stop`, `gameover` and `quit` are answered at once: the MCTS checks a stop
// flag between batches. A command that changes what the search reads stops
// a running search first, so every `go` gets exactly one `bestmove`.
//
// Budget: Sims simulations per move at most. With clock fields the move
// gets a target of remaining/30 + 3/4 of the increment (at most half the
// remaining time) + 0.9 x byoyomi. With a time model the target lowers the
// simulation count, so searches stay reproducible under load, and 1.5x the
// target is a hard stop; without one the target itself is the hard stop.
// No hard stop passes 0.8 x remaining + 0.95 x byoyomi.

#include "usi_mode.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "../movegen.h"
#include "../usi.h"
#include "../usi_engine.h"
#include "eval_client.h"
#include "mcts.h"
#include "selfplay.h"

namespace az {
namespace {

using Clock = std::chrono::steady_clock;

std::mutex outMu;

void say(const std::string& line) {
  std::lock_guard<std::mutex> lock(outMu);
  std::cout << line << '\n' << std::flush;
}

// ms = a + b * sims, least squares over the file's "<sims> <ms>" lines.
struct TimeModel {
  double a = 0, b = 0;
  bool ok = false;

  bool load(const std::string& path, std::string& err) {
    std::ifstream in(path);
    if (!in) {
      err = "cannot read " + path;
      return false;
    }
    std::vector<std::pair<double, double>> pts;
    for (std::string line; std::getline(in, line);) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream iss(line);
      double s, ms;
      if (iss >> s >> ms) pts.emplace_back(s, ms);
    }
    const double n = static_cast<double>(pts.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const auto& [x, y] : pts) {
      sx += x;
      sy += y;
      sxx += x * x;
      sxy += x * y;
    }
    const double det = n * sxx - sx * sx;
    if (pts.size() < 2 || det <= 0) {
      err = path + ": needs lines for two or more simulation counts";
      return false;
    }
    b = (n * sxy - sx * sy) / det;
    a = (sy - b * sx) / n;
    if (!(b > 0)) {
      err = path + ": time does not grow with simulations";
      return false;
    }
    ok = true;
    return true;
  }

  int sims(double ms) const {
    return static_cast<int>(std::clamp((ms - a) / b, 0.0, 1e9));
  }
};

struct Options {
  int sims = 1024;
  int batch = 16;
  int port = 51589;
  bool randomEval = false;
  size_t mateNodes = 5000;
  int probeDepth = 3;
  size_t probeNodes = 2000;
  float rootTemp = 1.f;
  chu::RuleSet rules = chu::RuleSet::Lishogi;
};

struct Go {
  std::string id;
  long long btime = -1, wtime = -1, binc = 0, winc = 0, byoyomi = 0,
            movetime = -1, nodes = -1;
  bool infinite = false;
};

Go parseGo(const std::vector<std::string>& tok) {
  Go g;
  g.id = chu::goId(tok);
  for (size_t i = 1; i < tok.size(); ++i) {
    const std::string& k = tok[i];
    if (k == "infinite") {
      g.infinite = true;
      continue;
    }
    long long* field = k == "btime"      ? &g.btime
                       : k == "wtime"    ? &g.wtime
                       : k == "binc"     ? &g.binc
                       : k == "winc"     ? &g.winc
                       : k == "byoyomi"  ? &g.byoyomi
                       : k == "movetime" ? &g.movetime
                       : k == "nodes"    ? &g.nodes
                                         : nullptr;
    if (field && i + 1 < tok.size())
      *field = std::strtoll(tok[++i].c_str(), nullptr, 10);
  }
  return g;
}

class Engine {
 public:
  explicit Engine(TimeModel tm) : tm_(tm) {}
  int run();

 private:
  void hello() const;
  void setOption(const std::string& name, const std::string& value);
  void connect();
  void search(Go g);
  void finishSearch(bool stop);

  TimeModel tm_;
  Options opt_;
  chu::Position pos_;
  std::vector<uint64_t> hashes_;  // canonical keys of the earlier positions
  bool havePos_ = false;
  bool afterGameover_ = false;
  std::unique_ptr<PipeEval> pipe_;
  RandomEval randomEval_;
  std::mt19937_64 rng_{0xC0FFEE};
  std::thread worker_;
  std::atomic<bool> searching_{false};
  std::atomic<bool> stop_{false};
};

void Engine::hello() const {
  say("id name chushogi-az");
  say("id author jh85");
  say("option name Sims type spin default 1024 min 1 max 1000000");
  say("option name Batch type spin default 16 min 1 max 256");
  say("option name EvalPort type spin default 51589 min 1 max 65535");
  say("option name Eval type combo default pipe var pipe var random");
  say("option name MateNodes type spin default 5000 min 0 max 100000000");
  say("option name ProbeDepth type spin default 3 min 0 max 64");
  say("option name ProbeNodes type spin default 2000 min 0 max 100000000");
  say("option name RootPolicyTemp type string default 1.0");
  say("option name Rules type combo default lishogi var lishogi var jcsa");
  say("usiok");
}

void Engine::setOption(const std::string& name, const std::string& value) {
  const long long v = std::strtoll(value.c_str(), nullptr, 10);
  auto clampInt = [&](long long lo, long long hi) {
    return static_cast<int>(std::clamp(v, lo, hi));
  };
  if (name == "Sims") {
    opt_.sims = clampInt(1, 1000000);
  } else if (name == "Batch") {
    opt_.batch = clampInt(1, 256);
  } else if (name == "EvalPort") {
    opt_.port = clampInt(1, 65535);
    pipe_.reset();
  } else if (name == "Eval") {
    opt_.randomEval = value == "random";
  } else if (name == "MateNodes") {
    opt_.mateNodes = static_cast<size_t>(std::max(0LL, v));
  } else if (name == "ProbeDepth") {
    opt_.probeDepth = clampInt(0, 64);
  } else if (name == "ProbeNodes") {
    opt_.probeNodes = static_cast<size_t>(std::max(0LL, v));
  } else if (name == "RootPolicyTemp") {
    const float t = std::strtof(value.c_str(), nullptr);
    if (std::isfinite(t) && t > 0.f) opt_.rootTemp = t;
    else say("info string error: RootPolicyTemp must be positive");
  } else if (name == "Rules") {
    opt_.rules = value == "jcsa" ? chu::RuleSet::JCSA : chu::RuleSet::Lishogi;
  }
}

// Connects to the eval server unless connected or using the random eval.
void Engine::connect() {
  if (opt_.randomEval || pipe_) return;
  try {
    pipe_ = std::make_unique<PipeEval>("127.0.0.1",
                                       static_cast<uint16_t>(opt_.port));
  } catch (const std::exception& e) {
    say("info string error: eval server on port " + std::to_string(opt_.port) +
        ": " + e.what());
  }
}

void Engine::finishSearch(bool stop) {
  if (stop) stop_ = true;
  if (worker_.joinable()) worker_.join();
}

void Engine::search(Go g) {
  const auto t0 = Clock::now();
  const std::string idTail = g.id.empty() ? "" : " id " + g.id;
  auto answer = [&](const std::string& move) {
    searching_ = false;  // before the bestmove, so the next go is accepted
    say("bestmove " + move + idTail);
  };
  if (!havePos_) {
    say("info string error: go without a valid position");
    return answer("resign");
  }
  const auto legal = chu::generateLegal(pos_);
  if (legal.empty()) return answer("resign");

  // Budget: simulations and a hard time limit (< 0: none).
  int sims = opt_.sims;
  double limitMs = -1;
  const bool sente = pos_.sideToMove == chu::Sente;
  const long long mine = sente ? g.btime : g.wtime;
  const long long inc = sente ? g.binc : g.winc;
  if (g.nodes > 0) {
    sims = static_cast<int>(std::min<long long>(g.nodes, INT_MAX / 2));
  } else if (g.infinite) {
    sims = INT_MAX / 2;
  } else if (g.movetime >= 0) {
    limitMs = static_cast<double>(g.movetime);
    if (tm_.ok) sims = std::clamp(tm_.sims(limitMs), 1, opt_.sims);
  } else if (mine >= 0 || g.byoyomi > 0) {
    const double rest = static_cast<double>(std::max(0LL, mine));
    const double target =
        std::min(rest / 30 + 0.75 * inc, rest / 2) + 0.9 * g.byoyomi;
    const double cap = 0.8 * rest + 0.95 * g.byoyomi;
    if (tm_.ok) {
      sims = std::clamp(tm_.sims(target), 1, opt_.sims);
      limitMs = std::min(1.5 * target, cap);
    } else {
      limitMs = std::min(target, cap);
    }
  }

  std::mutex m;
  std::condition_variable cv;
  bool finished = false;
  std::thread timer;
  if (limitMs >= 0) {
    const auto deadline =
        t0 + std::chrono::microseconds(std::llround(limitMs * 1000));
    timer = std::thread([&] {
      std::unique_lock<std::mutex> lock(m);
      if (!cv.wait_until(lock, deadline, [&] { return finished; }))
        stop_ = true;
    });
  }

  connect();
  Evaluator* eval = opt_.randomEval ? static_cast<Evaluator*>(&randomEval_)
                                    : static_cast<Evaluator*>(pipe_.get());
  std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
  chu::Move best = legal[pick(rng_)];  // if there is nothing to search with
  SearchResult res;
  bool searched = false;
  if (legal.size() > 1 && eval) {
    MctsConfig mc;
    mc.sims = sims;
    mc.batchSize = opt_.batch;
    mc.rootPolicyTemperature = opt_.rootTemp;
    mc.rootNoise = false;
    mc.rootMateNodes = opt_.mateNodes;
    mc.probeDepth = opt_.probeDepth;
    mc.probeNodes = opt_.probeNodes;
    mc.stop = &stop_;
    Mcts mcts(mc);
    const bool flip = pos_.sideToMove == chu::Gote;
    const chu::Position canon = flip ? pos_.flipped() : pos_;
    try {
      res = mcts.search(canon, hashes_, pos_.moveNumber - 1, *eval, rng_);
      searched = true;
      if (res.bestMove.from != chu::kNoSquare)
        best = flip ? chu::flipMove(res.bestMove) : res.bestMove;
    } catch (const std::exception& e) {
      say(std::string("info string error: ") + e.what() +
          "; playing a random move");
      pipe_.reset();  // reconnect on the next search
    }
  } else if (!eval) {
    say("info string error: no eval server; playing a random move");
  }

  {
    std::lock_guard<std::mutex> lock(m);
    finished = true;
  }
  cv.notify_all();
  if (timer.joinable()) timer.join();

  const std::string usi = chu::moveToUsi(best);
  if (searched) {
    const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             Clock::now() - t0).count();
    const double q = std::clamp(static_cast<double>(res.rootQ), 0.001, 0.999);
    const long cp = std::lround(400 * std::log10(q / (1 - q)));
    say("info depth 1 nodes " + std::to_string(res.rootVisits) + " time " +
        std::to_string(ms) + " score cp " + std::to_string(cp) + " pv " + usi);
  }
  answer(usi);
}

int Engine::run() {
  hello();
  for (std::string line; std::getline(std::cin, line);) {
    std::istringstream iss(line);
    std::vector<std::string> tok;
    for (std::string t; iss >> t;) tok.push_back(t);
    if (tok.empty()) continue;
    const std::string& c = tok[0];
    if (c == "quit") break;
    if (c == "usi") {
      hello();
    } else if (c == "isready") {
      if (!searching_) {
        finishSearch(false);
        connect();
      }
      say("readyok");
    } else if (c == "stop") {
      finishSearch(true);
    } else if (c == "go") {
      if (searching_) {
        say("info string error: go while searching; ignored");
        continue;
      }
      finishSearch(false);
      stop_ = false;
      searching_ = true;
      worker_ = std::thread(&Engine::search, this, parseGo(tok));
    } else if (c == "gameover") {
      finishSearch(true);
      afterGameover_ = true;
    } else if (c == "setoption" || c == "usinewgame" || c == "position") {
      finishSearch(true);  // these change what a search reads
      if (c == "setoption") {
        std::string name, value;
        if (chu::parseSetOption(tok, name, value)) setOption(name, value);
      } else if (c == "usinewgame") {
        afterGameover_ = false;
        havePos_ = false;
        hashes_.clear();
      } else {
        if (afterGameover_)
          say("info string error: position after gameover without usinewgame");
        std::vector<uint64_t> keys;
        std::string err;
        havePos_ = chu::parseUsiPosition(
            tok, opt_.rules, pos_, err,
            [&](const chu::Position& before, const chu::Move&) {
              keys.push_back(positionKey(
                  before.sideToMove == chu::Gote ? before.flipped() : before));
            });
        hashes_ = std::move(keys);
        if (!havePos_) say("info string error: " + err);
      }
    }
  }
  finishSearch(true);
  return 0;
}

}  // namespace

int runUsiMode(const std::string& timeModelPath) {
  TimeModel tm;
  std::string err;
  if (!timeModelPath.empty() && !tm.load(timeModelPath, err)) {
    std::cerr << "error: " << err << std::endl;
    return 1;
  }
  Engine engine(tm);
  return engine.run();
}

}  // namespace az
