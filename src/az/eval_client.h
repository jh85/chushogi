// Eval client: sends bit-packed positions to the Python eval server over
// TCP and receives policy logits + WDL (fp16 on the wire).
//
// Wire format (little-endian), matching az/eval_server.py:
//   request : u32 n | n * 1458 bytes packed planes | n * fp16 progress
//   response: u32 n | n * 50688 fp16 policy logits | n * 3 fp16 wdl

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

 private:
  int fd_ = -1;
  std::vector<uint8_t> buf_;  // reused
};

}  // namespace az
