/* C ABI over the chu shogi referee (position, legal moves, status, SFEN).
 *
 * No exceptions cross the boundary, no C++ types appear, and every buffer is
 * owned by the caller. A chu_pos is not thread-safe; use one per thread.
 * Built into libchushogi.a by `make libchushogi.a`.
 */
#ifndef CHUSHOGI_CAPI_H
#define CHUSHOGI_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct chu_pos chu_pos;

/* Return codes. */
#define CHU_OK 0
#define CHU_EILLEGAL (-1)  /* well-formed move that is not in the legal list */
#define CHU_ESYNTAX (-2)   /* unparseable SFEN or move, or bad enum value */
#define CHU_ESPACE (-3)    /* caller's buffer too small */
#define CHU_EINVAL (-4)    /* NULL argument */
#define CHU_EINTERNAL (-99) /* exception inside the engine */

/* Rule sets (chu_set_rules). */
#define CHU_LISHOGI 0
#define CHU_JCSA 1

/* Colours (chu_in_check, chu_status winner). */
#define CHU_SENTE 0
#define CHU_GOTE 1

/* Game-end status (chu_status). */
#define CHU_PLAYING 0
#define CHU_ROYALSLOST 1
#define CHU_BAREKING 2
#define CHU_STALEMATE 3
#define CHU_DRAW 4

/* Buffer sizes that always suffice. */
#define CHU_SFEN_CAP 512
#define CHU_LEGAL_CAP 16384

/* New position: the standard start position under lishogi rules, ply 0.
 * NULL on allocation failure. */
chu_pos* chu_new(void);
void chu_free(chu_pos* pos);

/* Sets the position from a full SFEN. Keeps the rule set, resets the ply
 * counter. On error the position is unchanged. */
int chu_set_sfen(chu_pos* pos, const char* sfen);

/* CHU_LISHOGI or CHU_JCSA; applies to the current position and later ones. */
int chu_set_rules(chu_pos* pos, int rules);

/* Applies one move. The token must be spelled exactly as chu_legal prints it
 * (so "8i8h=" is CHU_EILLEGAL even though the engine's parser accepts it). */
int chu_move(chu_pos* pos, const char* usi);

/* Writes the legal moves, space-separated, NUL-terminated. Returns the number
 * of moves, or CHU_ESPACE if cap is too small (CHU_LEGAL_CAP always fits). */
int chu_legal(const chu_pos* pos, char* buf, int cap);

/* Game-end status of the position under its rule set. If winner is not NULL
 * it receives CHU_SENTE, CHU_GOTE, or -1 (draw or still playing).
 * Repetition is history-dependent and not reported here. */
int chu_status(const chu_pos* pos, int* winner);

/* 1 if any royal of `side` is attacked, 0 if not. */
int chu_in_check(const chu_pos* pos, int side);

/* Writes the SFEN (board, side, lion-capture field, move number),
 * NUL-terminated. Returns its length, or CHU_ESPACE if cap is too small. */
int chu_sfen(const chu_pos* pos, char* buf, int cap);

/* Plies applied since chu_new or the last chu_set_sfen. */
int chu_ply(const chu_pos* pos);

#ifdef __cplusplus
}
#endif

#endif /* CHUSHOGI_CAPI_H */
