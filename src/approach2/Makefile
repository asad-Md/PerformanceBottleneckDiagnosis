CC    = gcc
ARCH  = $(shell uname -m | sed 's/x86_64/x86/' | sed 's/aarch64/arm64/')
CFLAGS  = -O2 -Wall
LDFLAGS = -lbpf -lelf -lz

.PHONY: all clean

all: pinner reader

pinner: pinner.c
	$(CC) $(CFLAGS) -o pinner pinner.c $(LDFLAGS)

reader: reader.c
	$(CC) $(CFLAGS) -o reader reader.c $(LDFLAGS)

# BPF object — compile with ecc (recommended) or clang directly
bpf:
	ecc perf_monitor.bpf.c

clean:
	rm -f pinner reader