CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

SRC := src/types.cpp src/position.cpp src/sfen.cpp src/usi.cpp src/movegen.cpp src/status.cpp src/mate.cpp src/main.cpp
BIN := chushogi-gen

GAMES1 ?= /data2/cs/chu_shogi_downloader/data/games
GAMES2 ?= /data2/cs/chu_shogi_downloader/data2/games

$(BIN): $(SRC) src/types.h src/position.h src/sfen.h src/usi.h src/movegen.h src/attacks.h src/status.h src/mate.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

# AlphaZero side: encoding inspection / MCTS / self-play binary.
AZSRC := src/types.cpp src/position.cpp src/sfen.cpp src/usi.cpp src/movegen.cpp src/status.cpp src/mate.cpp \
         src/az/encode.cpp src/az/mcts.cpp src/az/selfplay.cpp src/az/eval_client.cpp src/az/main.cpp
AZBIN := chushogi-az

$(AZBIN): $(AZSRC) src/types.h src/position.h src/sfen.h src/usi.h src/movegen.h src/attacks.h src/status.h src/mate.h src/az/encode.h src/az/mcts.h src/az/selfplay.h src/az/eval_client.h
	$(CXX) $(CXXFLAGS) -o $@ $(AZSRC)

PY ?= /data2/cs/venv/bin/python

az-test: $(BIN) $(AZBIN)
	$(PY) tests/az_encode_test.py $(GAMES1) $(GAMES2) --az ./$(AZBIN) --gen ./$(BIN)
	$(PY) tests/az_selfplay_test.py --az ./$(AZBIN) --gen ./$(BIN)
	$(PY) tests/az_mcts_test.py
	$(PY) tests/az_probe_test.py


# Regenerate the tracked test-case file from the downloaded game records.
# Explicit only: make test never rewrites it.
cases:
	python3 tests/make_cases.py tests/chushogi_cases.tsv $(GAMES1) $(GAMES2)

test: $(BIN)
	python3 tests/rule_test.py --binary ./$(BIN)
	python3 tests/jcsa_test.py --binary ./$(BIN)
	python3 tests/perft_test.py tests/perft_cases.tsv --binary ./$(BIN)
	python3 tests/status_test.py $(GAMES1) $(GAMES2) --binary ./$(BIN)
	python3 tests/mate_test.py $(GAMES1) $(GAMES2) --binary ./$(BIN)
	python3 tests/replay_test.py $(GAMES1) --binary ./$(BIN)
	python3 tests/replay_test.py $(GAMES2) --binary ./$(BIN)
	python3 tests/casefile_test.py tests/chushogi_cases.tsv --binary ./$(BIN)

# Play N random-versus-random games and write SFEN/USI records.
# Usage: make random-games GAMES=10 SEED=100
GAMES ?= 10
SEED ?= 1
random-games: $(BIN)
	python3 tools/random_match.py --games $(GAMES) --seed $(SEED) --binary ./$(BIN) --output random_games

clean:
	rm -f $(BIN)

.PHONY: test clean random-games cases
