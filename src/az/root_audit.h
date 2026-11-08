#pragma once
#include <string>
#include <vector>
#include "../position.h"
namespace az {
// CPU-only retrospective classification; recomputed bounded proofs are not
// evidence that the original search stored the same proof.
void auditRoot(const chu::Position& pos, const std::vector<std::string>& tokens);
}
