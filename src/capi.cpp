// C ABI over the referee: thin wrappers that catch every exception.

#include "capi.h"

#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

#include "attacks.h"
#include "movegen.h"
#include "sfen.h"
#include "status.h"
#include "usi.h"

struct chu_pos {
  chu::Position pos;
  int ply = 0;
};

namespace {

const char* const kStartSfen =
    "lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/"
    "3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1";

int copyOut(const std::string& s, char* buf, int cap) {
  if (cap < 0 || static_cast<size_t>(cap) < s.size() + 1) return CHU_ESPACE;
  std::memcpy(buf, s.c_str(), s.size() + 1);
  return CHU_OK;
}

}  // namespace

extern "C" {

chu_pos* chu_new(void) {
  try {
    chu_pos* p = new (std::nothrow) chu_pos;
    if (p && !chu::parseSfen(kStartSfen, p->pos)) {
      delete p;
      return nullptr;
    }
    return p;
  } catch (...) {
    return nullptr;
  }
}

void chu_free(chu_pos* pos) { delete pos; }

int chu_set_sfen(chu_pos* pos, const char* sfen) {
  if (!pos || !sfen) return CHU_EINVAL;
  try {
    chu::Position next = pos->pos;  // parseSfen keeps the rule set
    if (!chu::parseSfen(sfen, next)) return CHU_ESYNTAX;
    pos->pos = next;
    pos->ply = 0;
    return CHU_OK;
  } catch (...) {
    return CHU_EINTERNAL;
  }
}

int chu_set_rules(chu_pos* pos, int rules) {
  if (!pos) return CHU_EINVAL;
  if (rules != CHU_LISHOGI && rules != CHU_JCSA) return CHU_ESYNTAX;
  pos->pos.rules =
      rules == CHU_JCSA ? chu::RuleSet::JCSA : chu::RuleSet::Lishogi;
  return CHU_OK;
}

int chu_move(chu_pos* pos, const char* usi) {
  if (!pos || !usi) return CHU_EINVAL;
  try {
    const std::string token(usi);
    chu::Move m;
    if (!chu::parseUsiMove(token, m)) return CHU_ESYNTAX;
    if (chu::moveToUsi(m) != token || !chu::isLegal(pos->pos, m))
      return CHU_EILLEGAL;
    pos->pos.apply(m);
    ++pos->ply;
    return CHU_OK;
  } catch (const std::out_of_range&) {  // std::stoi on an absurd file number
    return CHU_ESYNTAX;
  } catch (...) {
    return CHU_EINTERNAL;
  }
}

int chu_legal(const chu_pos* pos, char* buf, int cap) {
  if (!pos || !buf) return CHU_EINVAL;
  try {
    const auto moves = chu::generateLegal(pos->pos);
    std::string out;
    for (const chu::Move& m : moves) {
      if (!out.empty()) out += ' ';
      out += chu::moveToUsi(m);
    }
    const int rc = copyOut(out, buf, cap);
    return rc == CHU_OK ? static_cast<int>(moves.size()) : rc;
  } catch (...) {
    return CHU_EINTERNAL;
  }
}

int chu_status(const chu_pos* pos, int* winner) {
  if (!pos) return CHU_EINVAL;
  try {
    const chu::StatusResult st = chu::evaluateStatus(pos->pos);
    if (winner) *winner = st.winner;
    switch (st.end) {
      case chu::GameEnd::Playing: return CHU_PLAYING;
      case chu::GameEnd::RoyalsLost: return CHU_ROYALSLOST;
      case chu::GameEnd::BareKing: return CHU_BAREKING;
      case chu::GameEnd::Stalemate: return CHU_STALEMATE;
      case chu::GameEnd::Draw: return CHU_DRAW;
    }
    return CHU_EINTERNAL;
  } catch (...) {
    return CHU_EINTERNAL;
  }
}

int chu_in_check(const chu_pos* pos, int side) {
  if (!pos) return CHU_EINVAL;
  if (side != CHU_SENTE && side != CHU_GOTE) return CHU_ESYNTAX;
  try {
    return chu::inCheck(pos->pos, side == CHU_SENTE ? chu::Sente : chu::Gote)
               ? 1
               : 0;
  } catch (...) {
    return CHU_EINTERNAL;
  }
}

int chu_sfen(const chu_pos* pos, char* buf, int cap) {
  if (!pos || !buf) return CHU_EINVAL;
  try {
    const std::string s = chu::toSfen(pos->pos);
    const int rc = copyOut(s, buf, cap);
    return rc == CHU_OK ? static_cast<int>(s.size()) : rc;
  } catch (...) {
    return CHU_EINTERNAL;
  }
}

int chu_ply(const chu_pos* pos) { return pos ? pos->ply : CHU_EINVAL; }

}  // extern "C"
