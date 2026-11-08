#include "root_audit.h"
#include "mcts.h"
#include "../movegen.h"
#include "../status.h"
#include "../usi.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>

namespace az {
void auditRoot(const chu::Position& actual, const std::vector<std::string>& t) {
  if (t.size()<4) throw std::runtime_error("audit-root played depth nodes [index:mass...]");
  const bool flip=actual.sideToMove==chu::Gote;
  const chu::Position pos=flip ? actual.flipped() : actual;
  chu::Move played;
  if (!chu::parseUsiMove(t[1],played)) throw std::runtime_error("bad audit move");
  if(flip) played=chu::flipMove(played);
  const int playedIndex=moveToIndex(played);
  std::map<int,float> policy;
  int mostVisited=-1;float maxMass=-1;
  for(size_t i=4;i<t.size();++i) {
    const auto colon=t[i].find(':');
    if(colon==std::string::npos) throw std::runtime_error("bad audit policy");
    const int index=std::stoi(t[i].substr(0,colon));
    const float mass=std::stof(t[i].substr(colon+1));
    policy[index]=mass;
    if(mass>maxMass){mostVisited=index;maxMass=mass;}
  }
  const auto legal=chu::generateLegal(pos);
  struct Verdict {bool hanging=false,lost=false;int plies=-1;};
  std::map<int,Verdict> verdicts;
  chu::MateSolver solver(1);solver.setMaxPly(std::stoi(t[2]));
  auto classify=[&](const chu::Move& move)->Verdict {
    const int index=moveToIndex(move);
    auto old=verdicts.find(index);if(old!=verdicts.end())return old->second;
    Verdict v;chu::Position child=pos;child.apply(move);
    const auto status=chu::evaluateStatus(child);
    if(status.end!=chu::GameEnd::Playing) {
      v.lost=status.winner==child.sideToMove;v.plies=v.lost?1:-1;
    } else {
      const int remaining=1001-actual.moveNumber;  // includes the played root move
      v.hanging=remaining>=2 && hangsLastRoyal(pos,move);
      if(v.hanging){v.lost=true;v.plies=2;}
      else if(std::stoi(t[2])>0 && remaining>=std::stoi(t[2])+3) {
        auto mate=solver.solve(child,std::stoull(t[3]));
        v.lost=mate.result==chu::MateResult::kMate;
        if(v.lost)v.plies=1+mate.matePly;
      }
    }
    verdicts[index]=v;return v;
  };
  bool playedLegal=false;
  bool substantialHang=false,substantialLoss=false;
  bool playedHang=false,playedLoss=false,bestHang=false,bestLoss=false;
  for(const auto& move:legal) {
    const int index=moveToIndex(move);
    if(index!=playedIndex && index!=mostVisited && policy[index]<.1f)continue;
    const auto v=classify(move);
    if(policy[index]>=.1f){substantialHang|=v.hanging;substantialLoss|=v.lost;}
    if(index==playedIndex){playedLegal=true;playedHang=v.hanging;playedLoss=v.lost;}
    if(index==mostVisited){bestHang=v.hanging;bestLoss=v.lost;}
  }
  if(!playedLegal)throw std::runtime_error("audit played move not legal");
  bool alternative=false;
  if(substantialHang || substantialLoss || playedLoss || bestLoss) {
    for(const auto& move:legal)if(!classify(move).lost){alternative=true;break;}
  }
  std::cout<<"audit {\"played_hanging\":"<<playedHang
    <<",\"played_reproved_loss\":"<<playedLoss<<",\"argmax_hanging\":"<<bestHang
    <<",\"argmax_reproved_loss\":"<<bestLoss<<",\"mass_ge_10_hanging\":"<<substantialHang
    <<",\"mass_ge_10_reproved_loss\":"<<substantialLoss
    <<",\"unproved_nonhanging_alternative\":"<<alternative<<"}\n"<<std::flush;
}
}
