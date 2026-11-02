#include <cmath>
#include <iostream>
#include <stdexcept>
#include "../src/az/selfplay.h"
#include "../src/usi.h"
void check(bool good,const char* message){if(!good)throw std::runtime_error(message);}
int main(){
  chu::Move best,other;check(chu::parseUsiMove("12i12h",best),"best move");
  check(chu::parseUsiMove("11i11h",other),"other move");
  az::SearchResult result;result.bestMove=best;
  result.policy={{az::moveToIndex(best),.75f},{az::moveToIndex(other),.25f}};
  const auto original=result.policy;
  for(auto condition: {std::pair<int,int>{20,19},{-1,20},{-1,999}}){
    std::mt19937_64 rng(123);int alternatives=0;
    for(int i=0;i<10000;++i)
      alternatives+=chu::moveToUsi(az::selectSelfplayMove(result,condition.first,condition.second,rng))=="11i11h";
    check(std::abs(alternatives/10000.f-.25f)<.02f,"sampling must remain proportional to visits, including late game");
  }
  for(auto condition:{std::pair<int,int>{20,20},{20,999},{0,0}}){
    std::mt19937_64 rng(17),copy=rng;
    for(int i=0;i<100;++i)check(chu::moveToUsi(az::selectSelfplayMove(result,condition.first,condition.second,rng))=="12i12h","cutoff must select best move");
    check(rng==copy,"greedy selection must not consume randomness");
  }
  check(result.policy==original,"sampling must not alter the policy training target");
  result.policy={{az::moveToIndex(best),1.f}};std::mt19937_64 rng(123);
  for(int i=0;i<100;++i)check(chu::moveToUsi(az::selectSelfplayMove(result,-1,999,rng))=="12i12h","one-hot proven mate must remain forced");
  std::cout<<"opening/full-game sampling, cutoff, visit probabilities, policy target and forced move checks passed\n";
}
