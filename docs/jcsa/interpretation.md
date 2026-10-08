# Interpretation notes (U1–U4 and the game-end items)

Our JCSA variant is the official over-the-board rules (2019 regulations with the 2004
supplement); relief measures for networked play are not applied. Each paragraph states
what the JCSA texts say, what is open, the reading we adopted, and the share of the
reachable-position count that the choice affects. Shares are static
shares of our candidate set of chu shogi boards (uniformly sampled; the reachable set is
over 99.5 % of it), measured on 2026-10-07.

## U1. Does the sakishishi ban also bind the enemy *lion*? — resolved

R-L4 says the opponent "may not take the lion back" without naming a piece type; R-L1
says adjacent lions may always be captured by a lion; H-3④ exempts tsukegui. **Decision:
the ban binds lions too, with two exceptions that take precedence: a lion may always
capture an adjacent lion, and a lion capture through tsukegui (taking the in-between
piece and the lion in one move) is permitted.** Under the lion-trading rules a lion can
capture a lion that has a foot only in those two ways, so the extension changes no
verdict in practice; the layer implements it literally. Effect: flag states only.

## U2. When is the "foot" evaluated? — resolved

R-L4 and H-3⑥ condition the ban on the lion to be taken back having a foot but do not say
whether this is judged at the moment of the capture or when the opponent tries to take the
lion; Okazaki's wording ("cannot be taken at once") reads as the latter. **Decision: at
the attempted recapture, with the recapturing piece lifted from its origin (X-rays through
it count), on the square of the lion to be taken.** Effect: flag states only.

## U3. Two lions on one side — resolved

With a lion and a promoted kirin, the texts speak of "our lion" in the singular.
**Decision: each lion is judged separately; a capture of lion X is banned iff X has a
foot.** Effect: flag states on boards with two lions on one side (small).

## U4. Does the ban cover every capture of the lion, or only taking back on the capture square? — resolved

JCSA says "take back"; lishogi bans captures on every square other than the one where the
lion was just captured and allows capturing the capturer there. **Decision: lishogi's
square semantics (ban on every square of the protected lion; the capturer may be taken on
the capture square), plus the hit-and-run coverage of row 2.** Effect: flag states only.

## Lance last-rank promotion (row 3, R-P4)

Officially only the pawn gets the last-rank second chance (R-P3); the official rules do
not codify the lance case explicitly. The JCSA commentary (中将棋対局規定 補足編, page
`rule_hosoku2004.htm`, dated 2004-06-06, section 1 on the non-promotion rules) explains
that the lance, a ranging piece that cannot retreat, was left unregulated because
entering unpromoted is pointless over the board; that the Federation's president
personally favours treating the lance like the pawn; and that the Federation applies
this only in networked play (通信対局) as a relief measure, a passage tied to chu shogi
software, i.e. computer-mediated play. The 2019 regulations codify the same restriction
(R-P5: correspondence games only). **Our JCSA variant follows the official over-the-board
rules: no lance relief.** lishogi's rule coincides with the relief measure. Effect on the
count: none (it only changes which moves are legal; a lance may promote on entering the
zone instead), and no proof game in our runs used a lance last-rank promotion
(0 of 2 000 + 41 444 replayed certificates).

## Bare king (row 4)

JCSA regulates only the ending "two kings and one piece other than a pawn or go-between"
(R-E4, R-E5), with the example of 九 excluding a piece the lone royal can recapture at
once (R-E6a); lishogi ends the game whenever a lone royal faces two or more counting
pieces, with adjacency and check conditions. **We keep lishogi's rule in the engine and
bound the difference.** Effect: terminal states and reachable boards, but only on boards
where one side is a lone royal: 1.9 × 10^-70 of the candidate set (exact count), so the
relative effect on the position count is below 10^-69.

## Stalemate (row 5)

lishogi: the stalemated side loses. JCSA: not regulated. **We keep lishogi's rule and bound
the difference.** Effect: terminal states and reachable boards; 0 stalemates among 98 998
sampled non-terminal candidate boards, i.e. a share below 3 × 10^-5 at the 3σ level.

## Leaving a royal capturable, and checkmate as the end of the game (row 6)

What the texts say. R-E1 gives two ways to win: checkmating the king ("詰める", defined on
the beginners' page as the king being in check with no square to escape to) **or**
capturing the king ("突き落とし"), and adds that announcing check is not required. The
list of fouls (R 六.2: moving a piece wrongly or promoting illegally, moving out of turn,
taking a move back, repeated touch-move violations) does not include leaving a royal in
check, and H-5 explains that the no-announcement and capture-wins clauses were adopted
precisely so that an unnoticed check loses by capture. **So leaving a royal capturable is
legal under JCSA, as under lishogi; it is not a foul.** Whether the game *ends* at
checkmate is open: R-E1 makes mate a win, and in practice play stops there, but the
capture is also a legal, winning move, so a mated player may be allowed to play on and be
captured.

The two readings and their size. Under the **capture reading** (ours: the game ends when
the last royal is captured) the JCSA and lishogi reachable sets are the same up to rows
1–5. Under the **mate reading** (a mated position is terminal) a large part of the state
space changes: in 98 998 sampled non-terminal candidate boards the side to move has its
royal attacked on 91.4 % and is mated in the JCSA sense (exactly one royal, attacked,
every legal move leaves a royal capturable) on **26.3 % ± 0.4 %**; and 11 of 12 sampled
proof games from our run pass through mated positions, typically hundreds of them. Under
the mate reading those 26 % of positions would be terminal, their continuations would be
unreachable, and almost every proof game would have to be re-planned to avoid mates. We
adopt the capture reading for the JCSA count ("JCSA move rules, game ends by royal
capture") and report the other as a sensitivity item: a complete one-ply retraction on
10 344 sampled boards on which the side not to move has a single, attacked royal that
would be mated on its turn found no un-mated predecessor for 3 117 of 39 593 sampled
boards, so at least 7.87 % (3σ interval 7.47–8.29 %) of the non-terminal positions would
be unreachable under the checkmate ending.
