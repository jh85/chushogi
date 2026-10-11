# JCSA rule specification package for `chushogi`

Prepared by the chu shogi position-count project (2026-10-07) from the rule pages of the
日本中将棋連盟 (JCSA): 中将棋連盟競技規則・総則編 2019年度版 (`rule2007.htm`, corrected
2019-09-14), 中将棋対局規定 補足編 2004-06-06 (`rule_hosoku2004.htm`), the beginners' page
(`rule.htm`), the piece page (`koma.htm`) and the historical Okazaki / 指南抄 page
(`Okazaki.htm`). Everything here is our paraphrase with citations; no original text or
images are included. Clause IDs: R-* = 2019 regulations (R-L = lion rules 七, R-P = promotion 四, R-E = victory 二, R-J = jitto 八), H-* = 2004
supplement (section.item), B-* = beginners' page, O-* = Okazaki, S-* = 指南抄.

Files:

- `jcsa-delta.md`: the differences from lishogi's rules (as implemented by `chushogi`)
  that affect move generation or game end, one row each, with the JCSA source.
- `interpretation.md`: the decided readings of the clauses the texts leave open (U1–U4,
  resolved 2026-10-08) and of the game-end items, one paragraph each, with the share of
  the position count that each reading affects.
- `cases.tsv`: 31 test positions built from the JCSA diagrams (29) and two played-through constructions (C17 / C17b, promoted kirin on the capture square) (piece patterns embedded in
  full legal positions with both royals and no stray attacks), each with its SFEN, side
  to move, counter-strike square, tested move and the expected verdict under lishogi and
  under JCSA. 7 rows are genuine differences; the rest are conformance checks. All
  lishogi verdicts were checked against `chushogi-gen` at commit 1b20c35.

SFEN conventions are lishogi's (`chushogi` engine): the third field is the counter-strike
square or `-`; promoted kirin / phoenix / drunk elephant are written as lion `n`, free
king `q`, king `k` (they are identical in every rule). Moves are USI (`7f6e`, double moves
`7f6e5d`, promotion `+`). A move written `a->b (any route)` means any legal lion double
move from a to b. Precedence between the sources: the 2019 regulations govern; the 2004
supplement is in force where the 2019 text is silent or consistent; the beginners' page is
informative; Okazaki and 指南抄 are historical and were superseded by the 2004 revision
(H-3 says the Federation moved from Okazaki's reading to the 全集/指南抄 reading).
