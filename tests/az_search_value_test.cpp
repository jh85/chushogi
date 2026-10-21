#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "../src/az/mcts.h"
#include "../src/sfen.h"
#include "../src/usi.h"

struct Eval : az::Evaluator {
  std::vector<std::vector<float>> inputs;
  void evaluate(int n, const float* x, float* p, float* v) override {
    chu::Move move;
    if (!chu::parseUsiMove("12i12h", move)) throw std::runtime_error("move");
    for (int i=0; i<n; ++i) {
      inputs.emplace_back(x+i*az::kNumPlanes*az::kBoard,
                          x+(i+1)*az::kNumPlanes*az::kBoard);
      std::fill(p+i*az::kPolicySize, p+(i+1)*az::kPolicySize, -100.f);
      p[i*az::kPolicySize+az::moveToIndex(move)] = 100.f;
      v[3*i]=std::log(.8f); v[3*i+1]=-100.f; v[3*i+2]=std::log(.2f);
    }
  }
};
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main() {
  chu::Position root;
  check(chu::parseSfen("lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1",root), "sfen");
  az::MctsConfig cfg; cfg.sims=1; cfg.batchSize=1; cfg.rootNoise=false;
  std::mt19937_64 rng(1); Eval eval;
  auto one=az::Mcts(cfg).search(root,{},0,eval,rng);
  check(std::abs(one.rootQ-.2f)<1e-5, "neural leaf value must be inverted and backed up");
  cfg.sims=2; Eval deeper;
  auto two=az::Mcts(cfg).search(root,{},0,deeper,rng);
  check(deeper.inputs.size()==3, "root plus two evaluated leaves expected");
  chu::Position expected=root;
  for (int i=0; i<3; ++i) {
    std::vector<float> planes(az::kNumPlanes*az::kBoard);
    az::encodePosition(expected,0,i,planes.data());
    check(planes==deeper.inputs[i], "search leaves must use canonical encoding and canonical policy indices");
    chu::Move move; check(chu::parseUsiMove(i==0 ? "12i12h" : "12d12e",move), "move");
    if (i<2) expected.apply(move);
  }
  check(std::abs(two.rootQ-.5f)<1e-5, "two-ply value perspective");
  cfg.sims=32; cfg.batchSize=8; Eval batched;
  auto many=az::Mcts(cfg).search(root,{},0,batched,rng);
  bool deep=false;
  for (const auto& x : batched.inputs) if(x[81*az::kBoard]>.0015f) deep=true;
  check(deep,"batched search must expand beyond evaluated leaves");
  check(many.rootVisits==32,"visit count");
  std::cout << "neural backup, canonical leaf/policy, multi-ply and batched expansion checks passed\n";
}
