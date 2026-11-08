#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "../src/az/selfplay.h"
#include "../src/movegen.h"
#include "../src/sfen.h"
#include "../src/usi.h"

void check(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
chu::Move move(const char* s){chu::Move m;check(chu::parseUsiMove(s,m),"move parse");return m;}
chu::Position position(const char* s){chu::Position p;check(chu::parseSfen(s,p),"SFEN parse");return p;}
struct ForcedEval:az::Evaluator {
  int index;
  explicit ForcedEval(const char* s):index(az::moveToIndex(move(s))){}
  void evaluate(int n,const float*,float* p,float* v)override{
    std::fill(p,p+size_t(n)*az::kPolicySize,-1000.f);
    for(int i=0;i<n;++i){p[size_t(i)*az::kPolicySize+index]=1000.f;v[3*i]=v[3*i+1]=v[3*i+2]=0;}
  }
};
int main(){
  auto p=position("4gtkt4/4se1h3f/f9m1/1c8o1/12/10s1/12/12/10+c1/2S5K3/12/2C5+r3 b - 319");
  az::MctsConfig cfg;cfg.sims=16;cfg.batchSize=16;cfg.rootNoise=false;cfg.probeDepth=3;cfg.collectStats=true;
  ForcedEval prior("4j5i");std::mt19937_64 rng(42);
  auto baseline=az::Mcts(cfg).search(p,{},318,prior,rng);
  check(chu::moveToUsi(baseline.bestMove)=="4j5i","fixture must reproduce the prior-driven blunder");
  check(baseline.unsafeRootMoves.empty(),"self-play exclusions default off");
  cfg.rootProofSelection=true;cfg.rootRoyalFilter=true;
  auto fixed=az::Mcts(cfg).search(p,{},318,prior,rng);
  check(!az::hangsLastRoyal(p,fixed.bestMove),"proof selection plus low-budget backstop");
  check(fixed.proofSelectionSkipped>0,"known proof must be used");
  check(fixed.policy==baseline.policy && fixed.rootVisits==baseline.rootVisits,"USI selection must not change visits or target");
  cfg.rootProofSelection=false;
  auto filterOnly=az::Mcts(cfg).search(p,{},318,prior,rng);
  check(!az::hangsLastRoyal(p,filterOnly.bestMove),"royal-only mode must also filter expanded proved children");
  cfg.rootProofSelection=true;cfg.probeDepth=0;
  auto backstop=az::Mcts(cfg).search(p,{},318,prior,rng);
  check(!az::hangsLastRoyal(p,backstop.bestMove),"backstop still works with probes disabled");
  cfg.safeSampling=cfg.safePolicy=true;
  auto cap=az::Mcts(cfg).search(p,{},999,prior,rng);
  check(cap.unsafeRootMoves.empty(),"known ply-cap draws must not be classified as future royal losses");
  cfg.safeSampling=cfg.safePolicy=false;cfg.probeDepth=3;
  int hangs=0;for(auto m:chu::generateLegal(p))hangs+=az::hangsLastRoyal(p,m);
  check(hangs==14,"fourteen immediate losses in rated-game position");
  auto prince=p;prince.board[0]=chu::makePiece(chu::Prince,chu::Sente);
  check(!az::hangsLastRoyal(prince,move("4j5i")),"losing a king is not losing the last royal when a prince survives");
  auto kingless=p;
  for(auto& piece:kingless.board) if(piece && chu::colorOf(piece)==chu::Sente && chu::isRoyal(chu::roleOf(piece))) piece=0;
  check(!az::hangsLastRoyal(kingless,move("10j10i")),"kingless tsume attacker is not a hanging-royal position");
  auto winning=position("12/6g5/7k4/12/12/12/12/12/12/12/12/7R3K b - 1");
  check(!az::hangsLastRoyal(winning,move("5l5c")),"capturing the opponent's last royal ends the game first");
  chu::Position two;two.board.fill(0);two.sideToMove=chu::Sente;
  two.board[chu::makeSq(4,5)]=chu::makePiece(chu::King,chu::Sente);
  two.board[chu::makeSq(4,6)]=chu::makePiece(chu::Prince,chu::Sente);
  two.board[chu::makeSq(9,11)]=chu::makePiece(chu::Gold,chu::Sente);
  two.board[chu::makeSq(3,5)]=chu::makePiece(chu::Lion,chu::Gote);
  two.board[chu::makeSq(0,0)]=chu::makePiece(chu::King,chu::Gote);
  check(az::hangsLastRoyal(two,move("10l10k")),"one lion reply can capture king and prince via mid and to");
  cfg.rootProofSelection=false;cfg.rootRoyalFilter=false;cfg.safeSampling=true;
  auto sampling=az::Mcts(cfg).search(p,{},318,prior,rng);
  check(sampling.policy==baseline.policy,"sampling-only flag leaves policy target unchanged");
  for(int i=0;i<1000;++i)check(!az::hangsLastRoyal(p,az::selectSelfplayMove(sampling,-1,318,rng,true)),"sampling exclusion, including zero eligible visit mass");
  check(!az::hangsLastRoyal(p,az::selectSelfplayMove(sampling,0,318,rng,true)),"post-temperature greedy exclusion");
  cfg.safeSampling=false;cfg.safePolicy=true;
  auto policy=az::Mcts(cfg).search(p,{},318,prior,rng);float sum=0;
  for(auto kv:policy.policy){sum+=kv.second;if(az::hangsLastRoyal(p,az::indexToMove(kv.first)))check(kv.second==0,"unsafe policy entry zeroed");}
  check(std::abs(sum-1)<1e-6,"filtered policy normalized even if no eligible visits");
  // With all moves proved lost, use the longest returned mate line.
  cfg.safePolicy=false;cfg.rootProofSelection=true;cfg.probeDepth=12;az::RandomEval uniform;
  auto lost=az::Mcts(cfg).search(p,{},318,uniform,rng);
  check(lost.allRootMovesLost,"all moves proved lost fixture");
  check(chu::moveToUsi(lost.bestMove)=="4j5j" && lost.selectedLossPlies==12,"prefer longer proof line over mate-in-one");
  cfg.safePolicy=true;cfg.safeSampling=true;
  auto allFiltered=az::Mcts(cfg).search(p,{},318,uniform,rng);
  check(allFiltered.policy==lost.policy && allFiltered.unsafeRootMoves.empty(),"all-lost target/sampling fallback remains usable");
  // The bad move has no immediate royal capture, but the opponent can force
  // a loss with a non-capturing check: the geometric backstop cannot catch it.
  auto deeper=position("5k6/12/12/12/12/12/12/12/10G1/g2r8/1K10/12 b - 1");
  ForcedEval deeperPrior("11k12l");
  check(!az::hangsLastRoyal(deeper,move("11k12l")),"deeper fixture must not hang immediately");
  cfg.safePolicy=cfg.safeSampling=false;cfg.probeDepth=3;cfg.rootProofSelection=false;
  auto raw=az::Mcts(cfg).search(deeper,{},0,deeperPrior,rng);
  check(chu::moveToUsi(raw.bestMove)=="11k12l","reproduce non-immediate proved-loss preference");
  cfg.rootProofSelection=true;cfg.rootRoyalFilter=true;
  auto safe=az::Mcts(cfg).search(deeper,{},0,deeperPrior,rng);
  check(chu::moveToUsi(safe.bestMove)!="11k12l" && !az::hangsLastRoyal(deeper,safe.bestMove),"proof-aware rejection beyond immediate capture");
  cfg.safeSampling=cfg.safePolicy=true;
  auto nearCap=az::Mcts(cfg).search(deeper,{},998,deeperPrior,rng);
  check(std::find(nearCap.unsafeRootMoves.begin(),nearCap.unsafeRootMoves.end(),az::moveToIndex(move("11k12l")))==nearCap.unsafeRootMoves.end(),"a tsume proof beyond the game cap is not a guaranteed loss");
  std::cout<<"root proof, mate length, backstop, multi-royal, sampling and policy checks passed\n";
}
