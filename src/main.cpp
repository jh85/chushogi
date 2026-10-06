// Chu Shogi legal move generator - command line driver.
//
// The program implements the pipeline
//   initial SFEN + ordered USI moves --(conversion layer)--> internal
//   Position --> legal move generator --> internal moves --(conversion
//   layer)--> USI move strings.
//
// Usage:
//   ./chushogi-gen                     reads commands from stdin (see below)
//   ./chushogi-gen "<sfen>" [moves...] prints legal moves for the position
//
// Stdin protocol (one command per line):
//   position sfen <board> <b|w> <lioncap|-> <num> [moves <m1> <m2> ...]
//       Set the position from an initial SFEN and replay the given ordered
//       USI moves. Every move is checked against the generated legal moves;
//       an illegal move is a fatal error.
//   legal   -> prints "legal <usi1> <usi2> ..." (all legal moves, USI format)
//   sfen    -> prints "sfen <board> <b|w> <lioncap|-> <num>" (current state)
//   status  -> prints "status <playing|royalslost|bareking|stalemate|draw>
//              [sente|gote]" (game-end evaluation of the current position)
//   seed <n>           -> seeds the random generator (for `go random`)
//   go random          -> prints "bestmove <usi>" with a uniformly random
//                         legal move ("bestmove resign" if there are none)
//   quit

#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "mate.h"
#include "movegen.h"
#include "attacks.h"
#include "sfen.h"
#include "status.h"
#include "usi.h"

namespace {

[[noreturn]] void fatal(const std::string& msg) {
  std::cerr << "error: " << msg << std::endl;
  std::exit(1);
}

// Applies one USI move through the conversion layer + generator.
void applyUsi(chu::Position& pos, const std::string& token) {
  chu::Move m;
  if (!chu::parseUsiMove(token, m)) fatal("unparseable USI move: " + token);
  if (!chu::isLegal(pos, m)) {
    fatal("illegal move " + token + " in position " + chu::toSfen(pos));
  }
  pos.apply(m);
}

// Stateful position handler: keeps the current position plus how it was
// reached, so a `position sfen S moves ...` command whose initial SFEN and
// move list extend the previous one only applies the new suffix (the players
// in a match are re-sent the start position plus the full move list every
// turn, so this keeps such sessions linear instead of quadratic).
class Session {
 public:
  // Parses "position sfen <f1> <f2> <f3> <f4> [moves ...]" tokens.
  void setPosition(const std::vector<std::string>& tok) {
    if (tok.size() < 6 || tok[1] != "sfen")
      fatal("expected: position sfen <board> <turn> <field3> <num> [moves ...]");
    std::string initial = tok[2] + " " + tok[3] + " " + tok[4] + " " + tok[5];
    std::vector<std::string> moves;
    size_t i = 6;
    if (i < tok.size()) {
      if (tok[i] != "moves") fatal("expected 'moves' in position command");
      for (++i; i < tok.size(); ++i) moves.push_back(tok[i]);
    }
    const bool extend =
        initial == initial_ && moves.size() >= moves_.size() &&
        std::equal(moves_.begin(), moves_.end(), moves.begin());
    if (!extend) {
      if (!chu::parseSfen(initial, pos_)) fatal("bad SFEN: " + initial);
      moves_.clear();
    }
    for (size_t k = moves_.size(); k < moves.size(); ++k)
      applyUsi(pos_, moves[k]);
    initial_ = initial;
    moves_ = std::move(moves);
  }

  void doMove(const std::string& token) {
    applyUsi(pos_, token);
    moves_.push_back(token);
  }

  const chu::Position& pos() const { return pos_; }

 private:
  chu::Position pos_;
  std::string initial_;
  std::vector<std::string> moves_;
};

void printLegal(const chu::Position& pos) {
  std::string line = "legal";
  for (const chu::Move& m : chu::generateLegal(pos)) {
    line += ' ';
    line += chu::moveToUsi(m);
  }
  std::cout << line << "\n" << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) {
    // Single-shot mode: chushogi-gen "<sfen>" [move1 move2 ...]
    chu::Position pos;
    if (!chu::parseSfen(argv[1], pos)) fatal("bad SFEN");
    for (int i = 2; i < argc; ++i) {
      std::istringstream iss(argv[i]);
      std::string tok;
      while (iss >> tok) applyUsi(pos, tok);
    }
    printLegal(pos);
    return 0;
  }

  Session session;
  std::mt19937_64 rng(std::random_device{}());
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream iss(line);
    std::vector<std::string> tok;
    for (std::string t; iss >> t;) tok.push_back(t);
    if (tok.empty()) continue;
    if (tok[0] == "quit") break;
    if (tok[0] == "position") {
      session.setPosition(tok);
      std::cout << "ok\n" << std::flush;
    } else if (tok[0] == "seed") {
      if (tok.size() != 2) fatal("expected: seed <n>");
      rng.seed(std::stoull(tok[1]));
      std::cout << "ok\n" << std::flush;
    } else if (tok[0] == "go") {
      if (tok.size() != 2 || tok[1] != "random") fatal("expected: go random");
      const chu::Position& pos = session.pos();
      const auto moves = chu::generateLegal(pos);
      if (moves.empty()) {
        std::cout << "bestmove resign\n" << std::flush;
      } else {
        std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
        std::cout << "bestmove " << chu::moveToUsi(moves[pick(rng)])
                  << "\n" << std::flush;
      }
    } else if (tok[0] == "legal") {
      printLegal(session.pos());
    } else if (tok[0] == "sfen") {
      std::cout << "sfen " << chu::toSfen(session.pos()) << "\n" << std::flush;
    } else if (tok[0] == "status") {
      chu::StatusResult st = chu::evaluateStatus(session.pos());
      std::cout << "status " << chu::gameEndName(st.end);
      if (st.winner >= 0) std::cout << ' ' << chu::colorName(st.winner);
      std::cout << "\n" << std::flush;
    } else if (tok[0] == "check") {
      // Is a side's royal (king/prince) under attack? Default: side to move.
      chu::Color c = session.pos().sideToMove;
      if (tok.size() > 1) {
        if (tok[1] == "sente") c = chu::Sente;
        else if (tok[1] == "gote") c = chu::Gote;
        else if (tok[1] != "us") fatal("expected: check [us|sente|gote]");
      }
      std::cout << "check " << (chu::inCheck(session.pos(), c) ? "yes" : "no")
                << "\n" << std::flush;
    } else if (tok[0] == "mate") {
      size_t limit = tok.size() > 1 ? std::stoull(tok[1]) : 300000;
      size_t ttMb = 32;
      for (size_t i = 2; i + 1 < tok.size(); ++i)
        if (tok[i] == "--tt-mb") ttMb = std::stoull(tok[i + 1]);
      chu::MateSolver solver(ttMb);
      if (tok.size() > 2 && tok[2] == "pndn") solver.setPnDnArithmetic(true);
      // optional depth cap (plies) as in the MCTS in-tree probes
      for (size_t i = 2; i + 1 < tok.size(); ++i) {
        if (tok[i] == "--depth") solver.setMaxPly(std::stoi(tok[i + 1]));
      }
      chu::MateAnswer ans = solver.solve(session.pos(), limit);
      const char* r = ans.result == chu::MateResult::kMate      ? "yes"
                      : ans.result == chu::MateResult::kNoMate  ? "no"
                                                                : "unknown";
      std::cout << "mate " << r << "\n";
      std::cout << "nodes " << ans.nodes << "\n";
      if (ans.result == chu::MateResult::kMate) {
        std::cout << "pv";
        for (const chu::Move& m : ans.pv) std::cout << ' ' << chu::moveToUsi(m);
        std::cout << "\n";
      }
      std::cout << std::flush;
    } else if (tok[0] == "move") {
      if (tok.size() != 2) fatal("expected: move <usi>");
      session.doMove(tok[1]);
      std::cout << "ok\n" << std::flush;
    } else {
      fatal("unknown command: " + tok[0]);
    }
  }
  return 0;
}
