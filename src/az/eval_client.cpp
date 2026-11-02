#include "eval_client.h"

#include <cstring>
#include <cstdlib>
#include <stdexcept>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include "encode.h"

namespace az {

namespace {

uint16_t fp16FromFloat(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  int exp = ((x >> 23) & 0xff) - 112;  // rebias to fp16
  uint32_t mant = x & 0x7fffff;
  if (exp <= 0) return static_cast<uint16_t>(sign);  // underflow to zero
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7bff);  // clamp
  return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

float fp16ToFloat(uint16_t h) {
  const uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  uint32_t f;
  if (exp == 0) {
    if (!mant) {
      f = sign;
    } else {  // subnormal
      exp = 0;
      while (!(mant & 0x400)) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3ff;
      f = sign | ((exp + 127 - 15 + 1) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000 | (mant << 13);  // inf/nan
  } else {
    f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

void writeAll(int fd, const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  while (n) {
    ssize_t w = ::write(fd, p, n);
    if (w <= 0) throw std::runtime_error("eval socket write failed");
    p += w;
    n -= w;
  }
}

void readAll(int fd, void* data, size_t n) {
  uint8_t* p = static_cast<uint8_t*>(data);
  while (n) {
    ssize_t r = ::read(fd, p, n);
    if (r <= 0) throw std::runtime_error("eval socket read failed");
    p += r;
    n -= r;
  }
}

}  // namespace

PipeEval::PipeEval(const std::string& host, uint16_t port) {
  fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd_ < 0) throw std::runtime_error("socket() failed");
  const int noDelay = 1;
  if (::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &noDelay, sizeof(noDelay)) != 0)
    throw std::runtime_error("TCP_NODELAY failed");
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  (void)host;  // v1: localhost only
  if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
    throw std::runtime_error("connect to eval server failed");
}

PipeEval::~PipeEval() {
  if (fd_ >= 0) ::close(fd_);
}

void PipeEval::evaluate(int n, const float* planesIn, float* policyOut,
                        float* wdlOut) {
  // request: u32 n | n*1458 packed planes | n fp16 progress
  buf_.resize(4 + size_t(n) * kPackedBytes + size_t(n) * 2);
  uint32_t hdr = static_cast<uint32_t>(n);
  std::memcpy(buf_.data(), &hdr, 4);
  uint8_t* pp = buf_.data() + 4;
  uint16_t* prog = reinterpret_cast<uint16_t*>(pp + size_t(n) * kPackedBytes);
  for (int i = 0; i < n; ++i) {
    const float* planes = planesIn + size_t(i) * kNumPlanes * kBoard;
    const float progress = packPlanes(planes, pp + size_t(i) * kPackedBytes);
    prog[i] = fp16FromFloat(progress);
  }
  writeAll(fd_, buf_.data(), buf_.size());

  uint32_t rn;
  readAll(fd_, &rn, 4);
  if (static_cast<int>(rn) != n)
    throw std::runtime_error("eval response count mismatch");
  const size_t polBytes = size_t(n) * kPolicySize * 2;
  const size_t wdlBytes = size_t(n) * 3 * 2;
  buf_.resize(polBytes + wdlBytes);
  readAll(fd_, buf_.data(), polBytes + wdlBytes);
  const uint16_t* pol = reinterpret_cast<const uint16_t*>(buf_.data());
  const uint16_t* wd = reinterpret_cast<const uint16_t*>(buf_.data() + polBytes);
  for (size_t i = 0; i < size_t(n) * kPolicySize; ++i)
    policyOut[i] = fp16ToFloat(pol[i]);
  for (size_t i = 0; i < size_t(n) * 3; ++i) wdlOut[i] = fp16ToFloat(wd[i]);
}

void PipeEval::evaluateLegal(int n, const float* x,
    const std::vector<std::vector<int>>& indices, float* p, float* v) {
  const char* mode = std::getenv("CHUSHOGI_EVAL_POLICY");
  if (mode && std::string(mode) == "full") { evaluate(n,x,p,v); return; }
  if (mode && std::string(mode) != "legal")
    throw std::runtime_error("CHUSHOGI_EVAL_POLICY must be full or legal");
  if (n < 1 || n > 1024 || indices.size() != size_t(n))
    throw std::runtime_error("invalid legal evaluation batch");
  size_t count = 0;
  for (const auto& row : indices) {
    if (row.size() > kPolicySize) throw std::runtime_error("too many policy indices");
    for (int index : row)
      if (index < 0 || index >= kPolicySize) throw std::runtime_error("invalid policy index");
    count += row.size();
  }
  const size_t base = 4 + size_t(n)*(kPackedBytes+2);
  buf_.resize(base + size_t(n)*2 + count*2);
  const uint32_t header = uint32_t(n) | 0x80000000u;
  std::memcpy(buf_.data(), &header, 4);
  for (int i=0;i<n;++i) {
    const float progress = packPlanes(x+size_t(i)*kNumPlanes*kBoard,
                                     buf_.data()+4+size_t(i)*kPackedBytes);
    const uint16_t half = fp16FromFloat(progress);
    std::memcpy(buf_.data()+4+size_t(n)*kPackedBytes+size_t(i)*2,&half,2);
    const uint16_t length = uint16_t(indices[i].size());
    std::memcpy(buf_.data()+base+size_t(i)*2,&length,2);
  }
  size_t offset = base+size_t(n)*2;
  for (const auto& row : indices) for (int index : row) {
    const uint16_t value = uint16_t(index);
    std::memcpy(buf_.data()+offset,&value,2); offset+=2;
  }
  writeAll(fd_,buf_.data(),buf_.size());
  uint32_t response;
  readAll(fd_,&response,4);
  if (response != header) throw std::runtime_error("legal eval response count/version mismatch");
  buf_.resize((count+size_t(n)*3)*2);
  readAll(fd_,buf_.data(),buf_.size());
  offset=0;
  for (int i=0;i<n;++i) for (int index : indices[i]) {
    uint16_t value; std::memcpy(&value,buf_.data()+offset,2); offset+=2;
    p[size_t(i)*kPolicySize+index]=fp16ToFloat(value);
  }
  for (int i=0;i<n*3;++i) {
    uint16_t value; std::memcpy(&value,buf_.data()+offset,2); offset+=2;
    v[i]=fp16ToFloat(value);
  }
}

}  // namespace az
