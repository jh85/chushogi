// chushogi-az: AlphaZero-side CLI. M1 surface: encoding inspection commands
// used by tests/az_encode_test.py.
//
//   position sfen <sfen> [moves <moves...>]   set position (same as main CLI)
//   encode                                     print the 82 planes as floats
//   legal-index                                legal moves as policy indices
//                                              (canonical flipped frame)
//   move-index <usi>...                        move -> index
//   index-move <idx>...                        index -> move (USI)
//   usi                                        switch to USI-Chu engine mode
//                                              (usi_mode.h); takes the
//                                              --time-model <file> argument

#include <algorithm>
#include <cmath>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <memory>
#include <random>

#include "../movegen.h"
#include "../position.h"
#include "../sfen.h"
#include "../usi.h"
#include "encode.h"
#include "eval_client.h"
#include "selfplay.h"
#include "usi_mode.h"

namespace {

[[noreturn]] void fatal(const std::string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(1);
}

// Analyze-only instrumentation; normal play uses the original evaluator.
struct TimedEval : az::Evaluator {
  az::Evaluator& base;
  double wallSeconds = 0, cpuSeconds = 0;
  uint64_t calls = 0, positions = 0;
  explicit TimedEval(az::Evaluator& evaluator) : base(evaluator) {}
  void evaluate(int n, const float* x, float* p, float* v) override {
    const auto start = std::chrono::steady_clock::now();
    const auto cpu = std::clock();
    base.evaluate(n,x,p,v);
    wallSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    cpuSeconds += double(std::clock()-cpu)/CLOCKS_PER_SEC;
    ++calls; positions += n;
  }
  void evaluateLegal(int n, const float* x, const std::vector<std::vector<int>>& indices,
                     float* p, float* v) override {
    const auto start = std::chrono::steady_clock::now();
    const auto cpu = std::clock();
    base.evaluateLegal(n,x,indices,p,v);
    wallSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    cpuSeconds += double(std::clock()-cpu)/CLOCKS_PER_SEC;
    ++calls; positions += n;
  }
};

constexpr const char* kInitialSfen =
    "lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/"
    "3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1";

}  // namespace

int main(int argc, char** argv) {
  std::string timeModel;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--time-model" && i + 1 < argc) timeModel = argv[++i];
    else fatal(std::string("unknown argument: ") + argv[i]);
  }
  chu::Position pos;
  if (!chu::parseSfen(kInitialSfen, pos)) fatal("bad initial SFEN");
  std::vector<uint64_t> hashes;         // canonical keys before current pos
  std::unique_ptr<az::PipeEval> pipeEval;
  std::mt19937_64 rng(0xC0FFEE);
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream iss(line);
    std::vector<std::string> tok;
    for (std::string t; iss >> t;) tok.push_back(t);
    if (tok.empty()) continue;
    if (tok[0] == "quit") break;
    if (tok[0] == "usi") return az::runUsiMode(timeModel);
    if (tok[0] == "selfplay") {
      // selfplay --games N --sims S --batch B --seed K --out DIR
      //          [--eval random|pipe] [--port N]
      int games = 1;
      int port = 51589;
      std::string evalKind = "random";
      az::SelfplayConfig cfg;
      std::string out = "az/data";
      for (size_t i = 1; i + 1 < tok.size(); i += 2) {
        if (tok[i] == "--games") games = std::stoi(tok[i + 1]);
        else if (tok[i] == "--sims") cfg.sims = std::stoi(tok[i + 1]);
        else if (tok[i] == "--batch") cfg.batchSize = std::stoi(tok[i + 1]);
        else if (tok[i] == "--temp-plies") cfg.tempPlies = std::stoi(tok[i + 1]);
        else if (tok[i] == "--seed") cfg.seed = std::stoull(tok[i + 1]);
        else if (tok[i] == "--out") out = tok[i + 1];
        else if (tok[i] == "--eval") evalKind = tok[i + 1];
        else if (tok[i] == "--port") port = std::stoi(tok[i + 1]);
        else if (tok[i] == "--mate-nodes") cfg.mcts.rootMateNodes = std::stoull(tok[i + 1]);
        else if (tok[i] == "--probe-depth") cfg.mcts.probeDepth = std::stoi(tok[i + 1]);
        else if (tok[i] == "--root-policy-temp") cfg.mcts.rootPolicyTemperature = std::stof(tok[i + 1]);
        else if (tok[i] == "--probe-nodes") cfg.mcts.probeNodes = std::stoull(tok[i + 1]);
        else fatal("unknown selfplay option: " + tok[i]);
      }
      if (cfg.tempPlies < -1) fatal("--temp-plies must be -1 (all moves) or nonnegative");
      cfg.mcts.sims = cfg.sims;
      cfg.mcts.batchSize = cfg.batchSize;
      std::string mkdir = "mkdir -p " + out;
      if (std::system(mkdir.c_str()) != 0) fatal("mkdir failed");
      az::RandomEval randomEval;
      std::unique_ptr<az::PipeEval> pipeEval;
      if (evalKind == "pipe")
        pipeEval = std::make_unique<az::PipeEval>("127.0.0.1",
                                                  static_cast<uint16_t>(port));
      else if (evalKind != "random")
        fatal("unknown --eval: " + evalKind);
      az::Evaluator* evalp = pipeEval ? static_cast<az::Evaluator*>(pipeEval.get()) : static_cast<az::Evaluator*>(&randomEval);
      for (int g = 0; g < games; ++g) {
        cfg.seed += 1;
        az::GameRecord rec = az::playGame(kInitialSfen, cfg, *evalp);
        char name[256];
        std::snprintf(name, sizeof(name), "%s/game_%ld_%04d.json", out.c_str(),
                      static_cast<long>(::getpid()), g);
        az::writeGameRecord(rec, name);
        std::cout << "game " << g << ": " << rec.moves.size() << " plies, "
                  << rec.reason << ", result " << rec.result << "\n"
                  << std::flush;
      }
      continue;
    }
    if (tok[0] == "position") {
      if (tok.size() < 3 || tok[1] != "sfen") fatal("expected: position sfen");
      std::string sfen = tok[2] + " " + tok[3] + " " + tok[4] + " " + tok[5];
      if (!chu::parseSfen(sfen, pos)) fatal("bad SFEN");
      hashes.clear();
      if (tok.size() > 6) {
        if (tok[6] != "moves") fatal("expected: moves");
        for (size_t i = 7; i < tok.size(); ++i) {
          chu::Move m;
          if (!chu::parseUsiMove(tok[i], m)) fatal("bad move: " + tok[i]);
          const chu::Position canon =
              pos.sideToMove == chu::Gote ? pos.flipped() : pos;
          hashes.push_back(az::positionKey(canon));
          pos.apply(m);
        }
      }
      std::cout << "ok\n" << std::flush;
    } else if (tok[0] == "seed") {  // seed <n>: reseed the RNG (match play)
      if (tok.size() != 2) fatal("expected: seed <n>");
      rng.seed(std::stoull(tok[1]));
      std::cout << "ok\n" << std::flush;
    } else if (tok[0] == "eval-server") {  // eval-server --port N
      int port = 51589;
      for (size_t i = 1; i + 1 < tok.size(); i += 2)
        if (tok[i] == "--port") port = std::stoi(tok[i + 1]);
      pipeEval = std::make_unique<az::PipeEval>("127.0.0.1",
                                                static_cast<uint16_t>(port));
      std::cout << "ok\n" << std::flush;
    } else if (tok[0] == "go" || tok[0] == "analyze") {
      const bool analyze = tok[0] == "analyze";
      if (tok.size() < 2 || tok[1] != "az") fatal("expected: go az --sims N");
      int sims = 200;
      size_t mateNodes = 0;
      size_t probeNodes = 2000;
      int probeDepth = 0;
      int tempPlies = 0;  // sample from the visit distribution this many plies
      int repetitionOccurrences = 2;
      int batchSize = 16;
      float rootPolicyTemperature = 1.f;
      std::string evalKind = "pipe";
      for (size_t i = 2; i + 1 < tok.size(); i += 2) {
        if (tok[i] == "--sims") sims = std::stoi(tok[i + 1]);
        else if (tok[i] == "--eval") evalKind = tok[i + 1];
        else if (tok[i] == "--mate-nodes") mateNodes = std::stoull(tok[i + 1]);
        else if (tok[i] == "--probe-nodes") probeNodes = std::stoull(tok[i + 1]);
        else if (tok[i] == "--probe-depth") probeDepth = std::stoi(tok[i + 1]);
        else if (tok[i] == "--temp-plies") tempPlies = std::stoi(tok[i + 1]);
        else if (tok[i] == "--batch") batchSize = std::stoi(tok[i + 1]);
        else if (tok[i] == "--root-policy-temp") rootPolicyTemperature = std::stof(tok[i + 1]);
        else if (tok[i] == "--repetition") {
          if (tok[i + 1] == "early") repetitionOccurrences = 2;
          else if (tok[i + 1] == "fourfold") repetitionOccurrences = 4;
          else fatal("--repetition must be early or fourfold");
        }
      }
      if (sims <= 0 || batchSize <= 0) fatal("sims and batch must be positive");
      if (!std::isfinite(rootPolicyTemperature) || rootPolicyTemperature <= 0.f)
        fatal("root policy temperature must be finite and positive");
      if (evalKind != "random" && !pipeEval)
        fatal("go az needs eval-server first");
      az::RandomEval randomEval;
      az::Evaluator* evalp =
          evalKind == "random"
              ? static_cast<az::Evaluator*>(&randomEval)
              : static_cast<az::Evaluator*>(pipeEval.get());
      az::MctsConfig mc;
      mc.sims = sims;
      mc.batchSize = batchSize;
      mc.rootPolicyTemperature = rootPolicyTemperature;
      mc.repetitionOccurrences = repetitionOccurrences;
      mc.collectStats = analyze;
      mc.rootNoise = false;
      mc.rootMateNodes = mateNodes;
      mc.probeDepth = probeDepth;
      mc.probeNodes = probeNodes;
      az::Mcts mcts(mc);
      const bool flip = pos.sideToMove == chu::Gote;
      const chu::Position canon = flip ? pos.flipped() : pos;
      const int plies = static_cast<int>(hashes.size());
      TimedEval measured(*evalp);
      const auto searchStart = std::chrono::steady_clock::now();
      const auto searchCpu = std::clock();
      az::SearchResult res = mcts.search(canon, hashes, plies,
          analyze ? static_cast<az::Evaluator&>(measured) : *evalp, rng);
      const double searchWall = std::chrono::duration<double>(std::chrono::steady_clock::now()-searchStart).count();
      const double searchCpuSeconds = double(std::clock()-searchCpu)/CLOCKS_PER_SEC;
      if (analyze) {
        const int repetitions = static_cast<int>(std::count(
            hashes.begin(), hashes.end(), az::positionKey(canon)));
        std::cout << "analysis {\"root_score\":" << res.rootQ
                  << ",\"search_wall_seconds\":" << searchWall
                  << ",\"search_cpu_seconds\":" << searchCpuSeconds
                  << ",\"evaluator_wall_seconds\":" << measured.wallSeconds
                  << ",\"evaluator_cpu_seconds\":" << measured.cpuSeconds
                  << ",\"evaluator_calls\":" << measured.calls
                  << ",\"evaluator_positions\":" << measured.positions
                  << ",\"root_policy_temperature\":" << mc.rootPolicyTemperature
                  << ",\"mate_nodes\":" << mc.rootMateNodes
                  << ",\"probe_depth\":" << mc.probeDepth
                  << ",\"probe_nodes\":" << mc.probeNodes
                  << ",\"proven_mate\":" << (res.provenMate ? "true" : "false")
                  << ",\"root_value\":" << res.rootValue
                  << ",\"root_wdl\":[" << res.rootWdl[0] << ","
                  << res.rootWdl[1] << "," << res.rootWdl[2] << "]"
                  << ",\"visits\":" << res.rootVisits
                  << ",\"root_prior_occurrences\":" << repetitions
                  << ",\"moves\":[";
        for (size_t i = 0; i < res.rootMoves.size(); ++i) {
          const auto& m = res.rootMoves[i];
          std::cout << (i ? "," : "") << "{\"move\":\""
                    << chu::moveToUsi(flip ? chu::flipMove(m.move) : m.move)
                    << "\",\"visits\":" << m.visits
                    << ",\"prior\":" << m.prior << ",\"q\":";
          if (m.visits) std::cout << m.q;
          else std::cout << "null";
          std::cout << ",\"repetition_visits\":" << m.repetitionVisits
                    << ",\"direct_repetition_draw\":"
                    << (m.directRepetitionDraw ? "true" : "false")
                    << ",\"child_prior_occurrences\":" << m.childPriorOccurrences
                    << "}";
        }
        std::cout << "]}\n" << std::flush;
        continue;
      }
      const auto legal = chu::generateLegal(pos);
      if (legal.empty()) {
        std::cout << "bestmove resign\n" << std::flush;
        continue;
      }
      chu::Move best = res.bestMove;
      if (best.from == chu::kNoSquare) {  // search found nothing: fall back
        best = legal[0];
      }
      if (plies < tempPlies && !res.policy.empty()) {
        // Match-play temperature: sample from the visit distribution (in the
        // canonical frame, like bestMove) so games between nets diversify.
        std::uniform_real_distribution<float> uni(0.f, 1.f);
        const float pick = uni(rng);
        float acc = 0.f;
        for (const auto& kv : res.policy) {
          acc += kv.second;
          if (pick <= acc + 1e-9f) {
            best = az::indexToMove(kv.first);
            break;
          }
        }
      }
      if (flip) best = chu::flipMove(best);
      std::cout << "bestmove " << chu::moveToUsi(best) << "\n" << std::flush;
    } else if (tok[0] == "sfen") {
      std::cout << "sfen " << chu::toSfen(pos) << "\n" << std::flush;
    } else if (tok[0] == "encode") {      // moveNumber counts plies + 1
      const chu::Position& p = pos;
      const int ply = p.moveNumber - 1;
      std::vector<float> planes(az::kNumPlanes * az::kBoard);
      az::encodePosition(p, 0, ply, planes.data());
      std::cout << "planes";
      for (float v : planes) std::cout << ' ' << v;
      std::cout << "\n" << std::flush;
    } else if (tok[0] == "legal-index") {
      const bool flip = pos.sideToMove == chu::Gote;
      const chu::Position canon = flip ? pos.flipped() : pos;
      std::cout << "legal-index";
      for (const chu::Move& m : chu::generateLegal(canon))
        std::cout << ' ' << az::moveToIndex(m);
      std::cout << "\n" << std::flush;
    } else if (tok[0] == "move-index") {
      std::cout << "move-index";
      for (size_t i = 1; i < tok.size(); ++i) {
        chu::Move m;
        if (!chu::parseUsiMove(tok[i], m)) fatal("bad move: " + tok[i]);
        std::cout << ' ' << az::moveToIndex(m);
      }
      std::cout << "\n" << std::flush;
    } else if (tok[0] == "index-move") {
      std::cout << "index-move";
      for (size_t i = 1; i < tok.size(); ++i) {
        const int idx = std::stoi(tok[i]);
        if (idx < 0 || idx >= az::kPolicySize) fatal("bad index");
        std::cout << ' ' << chu::moveToUsi(az::indexToMove(idx));
      }
      std::cout << "\n" << std::flush;
    } else {
      fatal("unknown command: " + tok[0]);
    }
  }
  return 0;
}
