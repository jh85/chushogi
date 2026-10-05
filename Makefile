CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

SRC := src/types.cpp src/position.cpp src/sfen.cpp src/usi.cpp src/movegen.cpp src/status.cpp src/mate.cpp src/main.cpp
BIN := chushogi-gen

GAMES1 ?= /data2/cs/chu_shogi_downloader/data/games
GAMES2 ?= /data2/cs/chu_shogi_downloader/data2/games

$(BIN): $(SRC) src/types.h src/position.h src/sfen.h src/usi.h src/movegen.h src/attacks.h src/status.h src/mate.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

# Regenerate the test-case file from the downloaded game records.
tests/chushogi_cases.tsv: tests/make_cases.py
	python3 tests/make_cases.py $@ $(GAMES1) $(GAMES2)

test: $(BIN) tests/chushogi_cases.tsv
	python3 tests/rule_test.py --binary ./$(BIN)
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

.PHONY: test clean random-games
