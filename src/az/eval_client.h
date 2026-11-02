// Eval client: sends bit-packed positions to the Python eval server over
// TCP and receives policy logits + WDL (fp16 on the wire).
//
// Wire format (little-endian), matching az/eval_server.py:
//   request : u32 n | n * 1458 bytes packed planes | n * fp16 progress
//   response: u32 n | n * 50688 fp16 policy logits | n * 3 fp16 wdl
// Legal variant: high bit of n set in both directions. Request appends n u16
// counts followed by row-major u16 canonical policy indices; response replaces
// the dense policy with just the requested fp16 logits, in the same order.
// CHUSHOGI_EVAL_POLICY=full forces legacy requests (default: legal).

#pragma once

#include <cstdint>
#include <string>

#include "mcts.h"

namespace az {

class PipeEval : public Evaluator {
 public:
  PipeEval(const std::string& host, uint16_t port);
  ~PipeEval() override;

  void evaluate(int n, const float* planesIn, float* policyOut,
                float* wdlOut) override;

  void evaluateLegal(int n, const float* x, const std::vector<std::vector<int>>& indices,
                     float* p, float* v) override;
 private:
  int fd_ = -1;
  std::vector<uint8_t> buf_;  // reused
};

}  // namespace az
