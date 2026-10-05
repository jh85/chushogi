// chushogi-az: AlphaZero-side CLI. M1 surface: encoding inspection commands
// used by tests/az_encode_test.py.
//
//   position sfen <sfen> [moves <moves...>]   set position (same as main CLI)
//   encode                                     print the 82 planes as floats
//   legal-index                                legal moves as policy indices
//                                              (canonical flipped frame)
//   move-index <usi>...                        move -> index
//   index-move <idx>...                        index -> move (USI)

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#include "../movegen.h"
#include "../position.h"
#include "../sfen.h"
#include "../usi.h"
#include "encode.h"
#include "eval_client.h"
#include "selfplay.h"

namespace {

[[noreturn]] void fatal(const std::string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(1);
}

constexpr const char* kInitialSfen =
    "lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/"
    "3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1";

}  // namespace

int main() {
  chu::Position pos;
  if (!chu::parseSfen(kInitialSfen, pos)) fatal("bad initial SFEN");
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream iss(line);
    std::vector<std::string> tok;
    for (std::string t; iss >> t;) tok.push_back(t);
    if (tok.empty()) continue;
    if (tok[0] == "quit") break;
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
        else if (tok[i] == "--seed") cfg.seed = std::stoull(tok[i + 1]);
        else if (tok[i] == "--out") out = tok[i + 1];
        else if (tok[i] == "--eval") evalKind = tok[i + 1];
        else if (tok[i] == "--port") port = std::stoi(tok[i + 1]);
        else fatal("unknown selfplay option: " + tok[i]);
      }
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
        std::snprintf(name, sizeof(name), "%s/game_%04d.json", out.c_str(), g);
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
      if (tok.size() > 6) {
        if (tok[6] != "moves") fatal("expected: moves");
        for (size_t i = 7; i < tok.size(); ++i) {
          chu::Move m;
          if (!chu::parseUsiMove(tok[i], m)) fatal("bad move: " + tok[i]);
          pos.apply(m);
        }
      }
      std::cout << "ok\n" << std::flush;
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
