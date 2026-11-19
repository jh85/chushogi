#include <algorithm>
#include <iostream>
#include <stdexcept>
#include "../src/az/repetition_guard.h"
#include "../src/az/selfplay.h"
#include "../src/movegen.h"
#include "../src/sfen.h"
#include "../src/usi.h"

void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
chu::Move move(const char* text){chu::Move m;check(chu::parseUsiMove(text,m),"parse move");return m;}
chu::Position position(const char* text){chu::Position p;check(chu::parseSfen(text,p),"parse SFEN");return p;}
bool contains(const std::vector<chu::Move>& moves,const char* text){
  return std::any_of(moves.begin(),moves.end(),[&](const chu::Move& m){return chu::moveToUsi(m)==text;});
}
void append(az::RepetitionHistory& history,chu::Position& p,const chu::Move& m){
  check(chu::isLegal(p,m),"legal fixture move");history.append(p,m);p.apply(m);
}

int main(){
  const char* sfen="6k1s3/6e5/8c3/5+O4+M1/2p9/12/12/2P9/F1M7T1/1S8S1/6E5/6K5 b - 447";
  const char* cycle[]={"7d8c","6a5b","8c7d","5b6a"};
  auto initial=position(sfen);auto p=initial;
  az::RepetitionHistory history;history.start(p);
  for(int i=0;i<10;++i)append(history,p,move(cycle[i%4]));
  auto blocked=az::perpetualLossMoves(p,history);
  check(contains(blocked,"8c7d"),"must see the opponent's next reply completing our loss");
  check(!contains(blocked,"8c6d"),"preserve the nonrepeating mating continuation");
  chu::MateSolver solver;auto proof=solver.solve(p,5000);
  check(proof.result==chu::MateResult::kMate && chu::moveToUsi(proof.pv.front())=="8c7d",
        "reproduce the history-blind root mate override without a network");
  check(!az::repetitionSafeMateLine(p,history,proof.pv),"nominal mate PV actually loses to repetition");
  az::MctsConfig config;config.sims=16;config.batchSize=4;config.rootNoise=false;
  config.rootMateNodes=5000;config.collectStats=true;
  az::RandomEval evaluator;std::mt19937_64 rng(42);
  auto raw=az::Mcts(config).search(p,{},456,evaluator,rng);
  check(raw.provenMate && chu::moveToUsi(raw.bestMove)=="8c7d","legacy reproduces actual losing move");
  for(const auto& m:blocked)config.excludedRootPolicyIndices.push_back(az::moveToIndex(m));
  config.rootMateLineAllowed=[&](const std::vector<chu::Move>& pv){return az::repetitionSafeMateLine(p,history,pv);};
  auto fixed=az::Mcts(config).search(p,{},456,evaluator,rng);
  check(!fixed.provenMate && chu::moveToUsi(fixed.bestMove)!="8c7d","guard must outrank mate override and neural priors");
  for(auto entry:fixed.policy)check(entry.first!=az::moveToIndex(move("8c7d")),"forbidden root must receive no search visits");
  append(history,p,move("8c7d"));
  auto defended=az::perpetualLossMoves(p,history);
  check(!contains(defended,"5b6a"),"allow a reply winning by the opponent's perpetual attack");
  append(history,p,move("5b6a"));
  check(history.verdict().fourfold && history.verdict().loser==chu::Sente,"CSA outcome: Black loses");

  // An earlier quiet cycle resets the run: matching just the last cycle is
  // insufficient. The run must extend back to the earliest matching state.
  p=initial;history.start(p);
  for(auto text:{"7d7e","6a5a","7e7d","5a6a"})append(history,p,move(text));
  for(int i=0;i<8;++i)append(history,p,move(cycle[i%4]));
  check(history.verdict().fourfold && history.verdict().loser<0,"interrupted attack is a neutral repetition");

  // Swap colors and mirror all moves, as the USI/MCTS boundary does.
  p=initial.flipped();history.start(p);
  for(int i=0;i<10;++i)append(history,p,chu::flipMove(move(cycle[i%4])));
  const auto mirrored=chu::moveToUsi(chu::flipMove(move("8c7d")));
  check(contains(az::perpetualLossMoves(p,history),mirrored.c_str()),"White is protected symmetrically");
  append(history,p,chu::flipMove(move("8c7d")));
  append(history,p,chu::flipMove(move("5b6a")));
  check(history.verdict().loser==chu::Gote,"mirrored checker loses");

  // Ordinary repetitions are allowed, rather than rejecting all cycles.
  p=position("k11/g11/12/12/12/12/12/12/12/12/11G/11K b - 1");history.start(p);
  const char* neutral[]={"1l2l","12a11a","2l1l","11a12a"};
  for(int i=0;i<10;++i)append(history,p,move(neutral[i%4]));
  check(!contains(az::perpetualLossMoves(p,history),"2l1l"),"preserve normal repetition draws");
  for(int i=10;i<12;++i)append(history,p,move(neutral[i%4]));
  check(history.verdict().fourfold && history.verdict().loser<0,"neutral fourfold");
  // Lishogi also penalizes perpetual attacks on nonroyal pieces. The same
  // queen loop can chase a silver while neither king is ever checked.
  p=position("6s5/12/12/5Q6/12/12/12/12/12/12/12/k10K b - 1");history.start(p);
  for(int i=0;i<10;++i)append(history,p,move(cycle[i%4]));
  check(contains(az::perpetualLossMoves(p,history),"8c7d"),"Lishogi perpetual nonroyal chase");
  for(int i=10;i<12;++i)append(history,p,move(cycle[i%4]));
  check(history.verdict().loser==chu::Sente,"perpetual chaser loses under Lishogi rules");
  std::cout<<"perpetual guard: rated-game loss, reply lookahead, mate override, colors, interrupted attacks and neutral draws passed\n";
}
