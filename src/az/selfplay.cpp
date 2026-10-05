#include "selfplay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>

#include "../sfen.h"
#include "../status.h"
#include "../usi.h"

namespace az {

namespace {

const char* kB64 =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64(const std::vector<uint8_t>& data) {
  std::string out;
  for (size_t i = 0; i < data.size(); i += 3) {
    uint32_t v = data[i] << 16;
    if (i + 1 < data.size()) v |= data[i + 1] << 8;
    if (i + 2 < data.size()) v |= data[i + 2];
    out += kB64[(v >> 18) & 63];
    out += kB64[(v >> 12) & 63];
    out += i + 1 < data.size() ? kB64[(v >> 6) & 63] : '=';
    out += i + 2 < data.size() ? kB64[v & 63] : '=';
  }
  return out;
}

std::string jsonEscape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out;
}

}  // namespace

void RandomEval::evaluate(int n, const float*, float* policyOut,
                          float* wdlOut) {
  std::fill(policyOut, policyOut + size_t(n) * kPolicySize, 0.0f);
  for (int i = 0; i < n; ++i) {
    wdlOut[3 * i] = 1.f / 3;
    wdlOut[3 * i + 1] = 1.f / 3;
    wdlOut[3 * i + 2] = 1.f / 3;
  }
}

GameRecord playGame(const std::string& initialSfen, const SelfplayConfig& cfg,
                    Evaluator& eval) {
  chu::Position pos;
  if (!chu::parseSfen(initialSfen, pos)) {
    fprintf(stderr, "error: bad initial SFEN\n");
    return {};
  }
  std::mt19937_64 rng(cfg.seed);

  GameRecord rec;
  rec.initialSfen = initialSfen;

  std::vector<uint64_t> hashes;  // canonical keys of positions BEFORE now
  auto canonOf = [](const chu::Position& p) {
    return p.sideToMove == chu::Gote ? p.flipped() : p;
  };

  Mcts mcts(cfg.mcts);
  std::vector<float> planes(kNumPlanes * kBoard);

  while (true) {
    const chu::Position canon = canonOf(pos);
    const uint64_t curKey = positionKey(canon);
    // Terminal adjudication: engine status, fourfold repetition, ply cap.
    chu::StatusResult st = chu::evaluateStatus(pos);
    const int reps =
        static_cast<int>(std::count(hashes.begin(), hashes.end(), curKey));
    const int plies = static_cast<int>(rec.moves.size());
    if (st.end != chu::GameEnd::Playing || reps >= 3 ||
        plies >= cfg.mcts.maxPlyCap) {
      if (st.end != chu::GameEnd::Playing) {
        rec.reason = chu::gameEndName(st.end);
        rec.result = st.winner == chu::Sente ? 1 : (st.winner == chu::Gote ? -1 : 0);
      } else {
        rec.reason = reps >= 3 ? "fourfold" : "plycap";
        rec.result = 0;
      }
      break;
    }

    const int repCount = reps;  // occurrences before the current position
    SearchResult res = mcts.search(canon, hashes, plies, eval, rng);
    if (res.policy.empty()) {  // no legal moves (should be caught by status)
      rec.reason = "nomoves";
      rec.result = pos.sideToMove == chu::Sente ? -1 : 1;
      break;
    }

    // Record the sample.
    Sample smp;
    smp.ply = plies;
    smp.stm = pos.sideToMove;
    encodePlanes(canon, repCount, plies, planes.data());
    smp.packedPlanes.resize(kPackedBytes);
    smp.progress = packPlanes(planes.data(), smp.packedPlanes.data());
    smp.policy = res.policy;
    smp.rootQ = res.rootQ;
    smp.a0gb = res.a0gb;
    rec.samples.push_back(std::move(smp));

    // Choose the move: temperature sampling early, argmax later.
    chu::Move chosen = res.bestMove;
    if (plies < cfg.tempPlies) {
      std::uniform_real_distribution<float> uni(0.f, 1.f);
      const float pick = uni(rng);
      float acc = 0.f;
      for (const auto& kv : res.policy) {
        acc += kv.second;
        if (pick <= acc + 1e-9f) {
          chosen = indexToMove(kv.first);
          break;
        }
      }
    }
    rec.moves.push_back(chu::moveToUsi(
        pos.sideToMove == chu::Gote ? chu::flipMove(chosen) : chosen));
    hashes.push_back(curKey);
    pos.apply(pos.sideToMove == chu::Gote ? chu::flipMove(chosen) : chosen);
  }

  // Fill z targets.
  for (Sample& s : rec.samples) {
    if (rec.result == 0)
      s.z = 0.5f;
    else
      s.z = (s.stm == 0 && rec.result == 1) || (s.stm == 1 && rec.result == -1)
                ? 1.0f
                : 0.0f;
  }
  return rec;
}

void writeGameRecord(const GameRecord& rec, const std::string& path) {
  std::ofstream f(path);
  f << "{\"initial\": \"" << jsonEscape(rec.initialSfen) << "\",\n";
  f << " \"result\": " << rec.result << ", \"reason\": \""
    << jsonEscape(rec.reason) << "\",\n \"moves\": [";
  for (size_t i = 0; i < rec.moves.size(); ++i)
    f << (i ? ", " : "") << "\"" << rec.moves[i] << "\"";
  f << "],\n \"samples\": [\n";
  for (size_t i = 0; i < rec.samples.size(); ++i) {
    const Sample& s = rec.samples[i];
    f << "  {\"ply\": " << s.ply << ", \"stm\": " << s.stm
      << ", \"policy\": [";
    for (size_t j = 0; j < s.policy.size(); ++j)
      f << (j ? ", " : "") << "[" << s.policy[j].first << ", "
        << s.policy[j].second << "]";
    f << "], \"rootq\": " << s.rootQ << ", \"a0gb\": " << s.a0gb
      << ", \"z\": " << s.z << ", \"progress\": " << s.progress
      << ", \"planes\": \"" << base64(s.packedPlanes) << "\"}"
      << (i + 1 < rec.samples.size() ? ",\n" : "\n");
  }
  f << "]}\n";
}

}  // namespace az
