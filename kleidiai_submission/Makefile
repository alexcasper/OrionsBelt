# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0
#
# Makefile for the KleidiAI GDN micro-kernel test harness.
#
# Usage:
#   make            # build with native ISA (auto-detected)
#   make test       # build and run tests
#   make bench      # build and run micro-benchmark
#   make clean
#
# Cross-compile for specific ISA levels:
#   make MARCH="-march=armv8-a"           # NEON floor (e.g. Cortex-A57)
#   make MARCH="-march=armv8.2-a+simd"    # NEON (e.g. Cortex-A76)
#   make MARCH="-march=armv8.2-a+sve"     # SVE1
#   make MARCH="-march=armv9-a+sve2"      # SVE2 (e.g. Cortex-A720)

CC      ?= cc
CFLAGS  ?= -O3 -Wall -Wextra -std=c11
MARCH   ?=
LDFLAGS ?= -lm

SRCS = test_kai_gdn.c kai/ukernels/gdn/kai_gdn_cumdecay_f32_sve.c \
       kai/ukernels/gdn/kai_gdn_gated_scan_f32_sve.c \
       kai/ukernels/gdn/kai_gdn_causal_dwconv1d_f32_k4_sve.c \
       kai/ukernels/gdn/kai_gdn_gemv_f32_f32_f32_1x4_neon.c

OBJS = $(SRCS:.c=.o)
BIN  = test_kai_gdn

INC = -I .

# Auto-detect ISA if MARCH is not set
ifeq ($(MARCH),)
  FEATURES := $(shell grep -m1 '^Features' /proc/cpuinfo 2>/dev/null | cut -d: -f2)
  ifneq (,$(findstring sve2,$(FEATURES)))
    MARCH := -march=armv9-a+sve2+i8mm+bf16
  else ifneq (,$(findstring sve,$(FEATURES)))
    MARCH := -march=armv8.2-a+sve
  else ifneq (,$(findstring asimddp,$(FEATURES)))
    MARCH := -march=armv8.2-a+dotprod
  else ifneq (,$(findstring asimd,$(FEATURES)))
    MARCH := -march=armv8-a
  else
    MARCH :=
  endif
endif

override CFLAGS += $(MARCH) $(INC)

.PHONY: all test bench clean

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(BIN)
	./$(BIN)

# Benchmark harness
BENCH_SRCS = bench_kai_gdn.c kai/ukernels/gdn/kai_gdn_cumdecay_f32_sve.c \
             kai/ukernels/gdn/kai_gdn_gated_scan_f32_sve.c \
             kai/ukernels/gdn/kai_gdn_causal_dwconv1d_f32_k4_sve.c \
             kai/ukernels/gdn/kai_gdn_gemv_f32_f32_f32_1x4_neon.c
BENCH_OBJS = $(BENCH_SRCS:.c=.o)
BENCH_BIN  = bench_kai_gdn

bench: $(BENCH_BIN)
	./$(BENCH_BIN) --repeats 30

$(BENCH_BIN): $(BENCH_OBJS)
	$(CC) $(CFLAGS) $(BENCH_OBJS) -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(BIN) $(BENCH_OBJS) $(BENCH_BIN)
