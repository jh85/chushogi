// USI-Chu engine mode of chushogi-az (USI-CHU.md), entered with `usi`.
#pragma once

#include <string>

namespace az {

// Runs the USI loop on stdin/stdout until `quit` or EOF; returns the exit
// code. `timeModelPath` is a file of "<sims> <milliseconds>" lines measured
// on this machine ("" = none: clocks then only bound the fixed Sims budget).
int runUsiMode(const std::string& timeModelPath);

}  // namespace az
