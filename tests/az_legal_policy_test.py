"""Legal-logit protocol: ragged batches, legacy clients, and invalid indices."""
from pathlib import Path
import socket
import socketserver
import struct
import sys
import threading
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'az'))
from eval_server import Handler, PACKED, read_exact
from inference_batcher import InferenceBatcher

class ProtocolTest(unittest.TestCase):
    def test_mixed_protocols_and_rejected_indices(self):
        def evaluate(raw, progress, indices=None):
            policy = np.arange(50688, dtype=np.float32)[None,:] + progress[:,None]
            policy = policy.astype(np.float16)
            if indices is not None:
                out = np.zeros((len(indices),max(map(len,indices))),dtype=np.float16)
                for i,row in enumerate(indices):out[i,:len(row)] = policy[i,row]
                policy = out
            return policy, np.repeat(progress[:,None],3,axis=1).astype(np.float16)
        class TCP(socketserver.ThreadingTCPServer):
            allow_reuse_address=True
            daemon_threads=True
        with TCP(('127.0.0.1',0),Handler) as server:
            server.srv=None;server.tcp_nodelay=True
            server.batcher=InferenceBatcher(evaluate,128,5)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                def request(rows):
                    n=2;header=n if rows is None else n|0x80000000
                    raw=struct.pack('<I',header)+bytes(n*PACKED)+np.array([1,2],dtype='<f2').tobytes()
                    if rows is not None:
                        raw+=np.array(list(map(len,rows)),dtype='<u2').tobytes()
                        raw+=np.array([v for row in rows for v in row],dtype='<u2').tobytes()
                    with socket.create_connection(server.server_address,timeout=3) as sock:
                        sock.sendall(raw)
                        self.assertEqual(struct.unpack('<I',read_exact(sock,4))[0],header)
                        count=n*50688 if rows is None else sum(map(len,rows))
                        values=np.frombuffer(read_exact(sock,(count+6)*2),dtype='<f2')
                    return values[:count],values[count:]
                full,wdl=request(None)
                # Duplicate, unordered, boundary and empty rows retain their order.
                for rows in [[[50687,3,3,0],[1]], [[],[9,0]], [[],[]]]:
                    sparse,got_wdl=request(rows)
                    expected=np.concatenate([full.reshape(2,-1)[i,row] for i,row in enumerate(rows)])
                    self.assertEqual(sparse.tobytes(),expected.tobytes())
                    self.assertEqual(got_wdl.tobytes(),wdl.tobytes())
                with socket.create_connection(server.server_address,timeout=3) as sock:
                    sock.sendall(struct.pack('<I',0x80000001)+bytes(PACKED+2)+struct.pack('<HH',1,50688))
                    self.assertEqual(sock.recv(1),b'')
            finally:
                server.shutdown();thread.join();server.batcher.close()

if __name__=='__main__':unittest.main()
