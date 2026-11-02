#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
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
struct CycleEval : az::Evaluator {
  std::vector<std::vector<float>> inputs;
  void evaluate(int n, const float* x, float* p, float* v) override {
    for (int i=0; i<n; ++i) {
      inputs.emplace_back(x+i*az::kNumPlanes*az::kBoard,
                          x+(i+1)*az::kNumPlanes*az::kBoard);
      const int ply=std::lround(x[(i*az::kNumPlanes+81)*az::kBoard]*1000);
      chu::Move move;
      chu::parseUsiMove(ply%4<2 ? "9h9g" : "9g9h",move);
      std::fill(p+i*az::kPolicySize,p+(i+1)*az::kPolicySize,-100.f);
      p[i*az::kPolicySize+az::moveToIndex(move)]=100.f;
      v[3*i]=0.f;v[3*i+1]=-100.f;v[3*i+2]=0.f;
    }
  }
};
struct TemperatureEval : az::Evaluator {
  bool forceRoot = false;
  std::vector<std::vector<float>> inputs;
  void evaluate(int n, const float* x, float* p, float* v) override {
    chu::Move a,b;
    chu::parseUsiMove("12i12h",a); chu::parseUsiMove("11i11h",b);
    for(int i=0;i<n;++i) {
      inputs.emplace_back(x+i*az::kNumPlanes*az::kBoard,x+(i+1)*az::kNumPlanes*az::kBoard);
      std::fill(p+i*az::kPolicySize,p+(i+1)*az::kPolicySize,-1000.f);
      const bool root = x[(i*az::kNumPlanes+81)*az::kBoard]==0.f;
      p[i*az::kPolicySize+az::moveToIndex(a)] = forceRoot && root ? 100.f : std::log(4.f);
      p[i*az::kPolicySize+az::moveToIndex(b)] = forceRoot && root ? -1000.f : 0.f;
      v[3*i]=std::log(.8f);v[3*i+1]=-100.f;v[3*i+2]=std::log(.2f);
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
  check(std::abs(one.rootWdl[0]-.8f)<1e-5,"root diagnostic WDL");
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
  // One prior occurrence must cut off in legacy mode, but fourfold mode
  // must evaluate the leaf until three prior occurrences exist.
  chu::Position child = root;
  chu::Move pawn; check(chu::parseUsiMove("12i12h", pawn), "move");
  child.apply(pawn);
  const uint64_t childKey = az::positionKey(child.flipped());
  cfg.sims=1; cfg.batchSize=1; cfg.collectStats=true;
  for (int prior=1; prior<=3; ++prior) {
    std::vector<uint64_t> history(prior,childKey);
    cfg.repetitionOccurrences=2; Eval legacy;
    auto early=az::Mcts(cfg).search(root,history,0,legacy,rng);
    check(std::abs(early.rootQ-.5f)<1e-5,"legacy repetition shortcut");
    cfg.repetitionOccurrences=4; Eval exact;
    auto four=az::Mcts(cfg).search(root,history,0,exact,rng);
    check(std::abs(four.rootQ-(prior==3 ? .5f : .2f))<1e-5,
          "fourfold requires three prior occurrences");
    if (prior<3) {
      check(exact.inputs.size()==2,"nonterminal repeated leaf must be evaluated");
      check(exact.inputs[1][79*az::kBoard]==1.f,"first repetition input plane");
      check(exact.inputs[1][80*az::kBoard]==(prior==2 ? 1.f : 0.f),
            "second repetition input plane");
    }
    int repetitionVisits=0;
    for (const auto& move : four.rootMoves) repetitionVisits+=move.repetitionVisits;
    check(repetitionVisits==(prior==3 ? 1 : 0),"diagnostic repetition visits");
  }
  cfg.sims=12;cfg.repetitionOccurrences=4;CycleEval cycle;
  auto cyclic=az::Mcts(cfg).search(root,{},0,cycle,rng);
  check(cycle.inputs.size()==12,"fourfold search must include the current root in path history");
  check(cycle.inputs[4][79*az::kBoard]==1.f,"root's second occurrence reaches network");
  check(cycle.inputs[8][80*az::kBoard]==1.f,"root's third occurrence reaches network");
  int repetitions=0;
  for(const auto& move:cyclic.rootMoves) repetitions+=move.repetitionVisits;
  check(repetitions==1,"fourth root occurrence must terminate at depth twelve");
  // Root priors must be softmax(logits/T), with unchanged NN values.
  cfg=az::MctsConfig{};cfg.sims=1;cfg.batchSize=1;cfg.rootNoise=false;cfg.collectStats=true;
  for(float temperature : {1.f,1.25f}) {
    cfg.rootPolicyTemperature=temperature;TemperatureEval evaluator;
    auto result=az::Mcts(cfg).search(root,{},0,evaluator,rng);
    float pa=0,pb=0,sum=0;
    for(const auto& edge:result.rootMoves) {
      sum+=edge.prior;
      if(chu::moveToUsi(edge.move)=="12i12h") pa=edge.prior;
      if(chu::moveToUsi(edge.move)=="11i11h") pb=edge.prior;
    }
    check(std::abs(pa/pb-std::pow(4.f,1.f/temperature))<1e-5,"root temperature prior ratio");
    check(std::abs(sum-1.f)<1e-5,"root priors sum to one");
    check(std::abs(result.rootWdl[0]-.8f)<1e-5,"temperature must preserve neural WDL");
  }
  // Identical forced root priors isolate deeper-node behavior: changing T
  // must leave the complete evaluated-position sequence unchanged.
  cfg.sims=64;TemperatureEval cold,warm;cold.forceRoot=warm.forceRoot=true;
  cfg.rootPolicyTemperature=1.f;std::mt19937_64 coldRng(42),warmRng(42);
  auto coldResult=az::Mcts(cfg).search(root,{},0,cold,coldRng);
  cfg.rootPolicyTemperature=1.25f;
  auto warmResult=az::Mcts(cfg).search(root,{},0,warm,warmRng);
  check(cold.inputs==warm.inputs && coldResult.rootQ==warmResult.rootQ,"temperature must affect only root priors");
  for(float invalid : {0.f,-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
    cfg.rootPolicyTemperature=invalid;bool rejected=false;
    try { az::Mcts search(cfg); } catch(const std::invalid_argument&) { rejected=true; }
    check(rejected,"invalid root temperatures must be rejected");
  }
  std::cout << "neural backup, canonical inputs/policy, batched expansion, fourfold history, repetition planes and root temperature checks passed\n";
}
