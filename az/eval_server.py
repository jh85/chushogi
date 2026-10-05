#!/usr/bin/env python3
"""GPU eval server for AZ self-play: receives bit-packed positions from
chushogi-az workers over TCP, replies with policy logits + WDL.

Protocol (length-prefixed little-endian frames):
  request : u32 n | n * 1458 bytes (81 bit planes) | n * fp16 progress
  response: u32 n | n * 50688 fp16 policy logits | n * 3 fp16 wdl

--watch makes the server reload the checkpoint whenever the file changes
(the training loop publishes improved checkpoints atomically).
"""

import argparse
import os
import socket
import socketserver
import struct
import threading
import time

import numpy as np
import torch

import model as M

PACKED = 81 * M.BOARD // 8


class Server:
    def __init__(self, checkpoint, device, dtype):
        self.device = device
        self.dtype = dtype
        self.checkpoint = checkpoint
        self.lock = threading.Lock()
        self.net = M.ChuNet().to(device)
        self.mtime = None
        self.reload(force=True)

    def reload(self, force=False):
        if not self.checkpoint:
            return
        try:
            mt = os.path.getmtime(self.checkpoint)
        except OSError:
            return
        if not force and mt == self.mtime:
            return
        state = torch.load(self.checkpoint, map_location="cpu",
                           weights_only=True)
        self.net.load_state_dict(state["model"] if "model" in state else state)
        self.mtime = mt
        print(f"[eval] loaded {self.checkpoint} (step {state.get('step', '?')})",
              flush=True)

    def eval(self, packed, progress):
        planes = M.unpack_batch(packed, progress).to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype,
                                             enabled=self.device == "cuda"):
            policy, wdl = self.net(planes)
        return (policy.float().cpu().numpy().astype(np.float16),
                wdl.float().cpu().numpy().astype(np.float16))


def read_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf.extend(chunk)
    return bytes(buf)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        srv = self.server.srv
        while True:
            try:
                head = read_exact(self.request, 4)
                (n,) = struct.unpack("<I", head)
                packed = read_exact(self.request, n * PACKED)
                prog = np.frombuffer(read_exact(self.request, n * 2),
                                     dtype=np.float16)
            except ConnectionError:
                return
            if srv.watch:
                srv.reload()
            with srv.lock:
                policy, wdl = srv.eval(packed, prog.astype(np.float32))
            out = struct.pack("<I", n) + policy.tobytes() + wdl.tobytes()
            self.request.sendall(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--port", type=int, default=51589)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--watch", action="store_true",
                    help="reload checkpoint when the file changes")
    a = ap.parse_args()

    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    srv = Server(a.checkpoint, device, torch.bfloat16)
    srv.watch = a.watch

    class Server_(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server_(("127.0.0.1", a.port), Handler) as httpd:
        httpd.srv = srv
        print(f"[eval] serving on 127.0.0.1:{a.port} ({device})", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
