#!/usr/bin/env python3
import http.server
import socketserver
import sys

class COOPCOEPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 6060
    socketserver.TCPServer.allow_reuse_address = True
    print(f"Serving ESMFold1 WASM on http://localhost:{port} with COOP/COEP headers...")
    with socketserver.TCPServer(("", port), COOPCOEPHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()
