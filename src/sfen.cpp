#include "sfen.h"

#include <cctype>

#include "usi.h"

namespace chu {

namespace {

Role roleFromLetter(char c, bool promoted) {
  if (!promoted) {
    switch (c) {
      case 'p': return Pawn;
      case 'l': return Lance;
      case 'i': return GoBetween;
      case 'a': return Chariot;
      case 'm': return SideMover;
      case 'v': return VerticalMover;
      case 'c': return Copper;
      case 's': return Silver;
      case 'f': return Leopard;
      case 't': return Tiger;
      case 'g': return Gold;
      case 'e': return Elephant;
      case 'b': return Bishop;
      case 'r': return Rook;
      case 'h': return Horse;
      case 'd': return Dragon;
      case 'o': return Kirin;
      case 'x': return Phoenix;
      case 'k': return King;
      case 'n': return Lion;
      case 'q': return Queen;
      default: return RoleCount;
    }
  }
  switch (c) {
    case 'p': return Tokin;
    case 'l': return WhiteHorse;
    case 'i': return ElephantP;
    case 'a': return Whale;
    case 'm': return Boar;
    case 'v': return Ox;
    case 'c': return SideMoverP;
    case 's': return VerticalMoverP;
    case 'f': return BishopP;
    case 't': return Stag;
    case 'g': return RookP;
    case 'b': return HorseP;
    case 'r': return DragonP;
    case 'h': return Falcon;
    case 'd': return Eagle;
    case 'o': return LionP;
    case 'x': return QueenP;
    case 'e': return Prince;
    default: return RoleCount;
  }
}

// Returns "+x" style string for the role, or empty if not representable.
std::string letterFromRole(Role r) {
  switch (r) {
    case Pawn: return "p";
    case Lance: return "l";
    case GoBetween: return "i";
    case Chariot: return "a";
    case SideMover: return "m";
    case VerticalMover: return "v";
    case Copper: return "c";
    case Silver: return "s";
    case Leopard: return "f";
    case Tiger: return "t";
    case Gold: return "g";
    case Elephant: return "e";
    case Bishop: return "b";
    case Rook: return "r";
    case Horse: return "h";
    case Dragon: return "d";
    case Kirin: return "o";
    case Phoenix: return "x";
    case King: return "k";
    case Lion: return "n";
    case Queen: return "q";
    case Tokin: return "+p";
    case WhiteHorse: return "+l";
    case ElephantP: return "+i";
    case Whale: return "+a";
    case Boar: return "+m";
    case Ox: return "+v";
    case SideMoverP: return "+c";
    case VerticalMoverP: return "+s";
    case BishopP: return "+f";
    case Stag: return "+t";
    case RookP: return "+g";
    case HorseP: return "+b";
    case DragonP: return "+r";
    case Falcon: return "+h";
    case Eagle: return "+d";
    case LionP: return "+o";
    case QueenP: return "+x";
    case Prince: return "+e";
    default: return "";
  }
}

}  // namespace

bool parseSfen(const std::string& sfen, Position& out) {
  Position pos;
  pos.board.fill(0);

  size_t sp1 = sfen.find(' ');
  std::string board = sfen.substr(0, sp1);

  int x = kFiles - 1, y = 0;
  bool promoted = false;
  for (size_t i = 0; i < board.size(); ++i) {
    char c = board[i];
    if (c == '/') {
      if (promoted) return false;
      ++y;
      x = kFiles - 1;
      if (y >= kRanks) return false;
    } else if (c == '+') {
      if (promoted) return false;
      promoted = true;
    } else if (c >= '0' && c <= '9') {
      if (promoted) return false;
      int n = 0;
      while (i < board.size() && std::isdigit(board[i])) {
        n = n * 10 + (board[i] - '0');
        ++i;
      }
      --i;
      if (n < 1 || n > kFiles) return false;
      x -= n;
      if (x < -1) return false;
    } else {
      Role r = roleFromLetter(static_cast<char>(std::tolower(c)), promoted);
      if (r == RoleCount || !onBoard(x, y)) return false;
      pos.board[makeSq(x, y)] =
          makePiece(r, std::isupper(c) ? Sente : Gote);
      promoted = false;
      --x;
    }
  }
  if (promoted || y != kRanks - 1) return false;

  // Defaults for missing fields.
  std::string turn = "b", field3 = "-", num = "1";
  if (sp1 != std::string::npos) {
    size_t sp2 = sfen.find(' ', sp1 + 1);
    turn = sfen.substr(sp1 + 1, sp2 == std::string::npos
                                       ? std::string::npos
                                       : sp2 - sp1 - 1);
    if (sp2 != std::string::npos) {
      size_t sp3 = sfen.find(' ', sp2 + 1);
      field3 = sfen.substr(sp2 + 1, sp3 == std::string::npos
                                        ? std::string::npos
                                        : sp3 - sp2 - 1);
      if (sp3 != std::string::npos) num = sfen.substr(sp3 + 1);
    }
  }

  if (turn == "b") pos.sideToMove = Sente;
  else if (turn == "w") pos.sideToMove = Gote;
  else return false;

  if (field3 != "-") {
    int sq;
    if (!parseSquare(field3, sq)) return false;
    pos.lastLionCapture = static_cast<int16_t>(sq);
  }

  try {
    pos.moveNumber = std::stoi(num);
  } catch (...) {
    return false;
  }

  pos.rules = out.rules;  // not part of the SFEN
  out = pos;
  return true;
}

std::string toSfen(const Position& pos) {
  std::string s;
  for (int y = 0; y < kRanks; ++y) {
    int emptyRun = 0;
    for (int x = kFiles - 1; x >= 0; --x) {
      uint8_t p = pos.board[makeSq(x, y)];
      if (!p) {
        ++emptyRun;
        continue;
      }
      if (emptyRun) {
        s += std::to_string(emptyRun);
        emptyRun = 0;
      }
      std::string letter = letterFromRole(roleOf(p));
      if (colorOf(p) == Sente)
        for (char& c : letter) c = static_cast<char>(std::toupper(c));
      s += letter;
    }
    if (emptyRun) s += std::to_string(emptyRun);
    if (y < kRanks - 1) s += '/';
  }
  s += pos.sideToMove == Sente ? " b " : " w ";
  s += pos.lastLionCapture == kNoSquare ? "-" : squareKey(pos.lastLionCapture);
  s += ' ';
  s += std::to_string(pos.moveNumber);
  return s;
}

}  // namespace chu
