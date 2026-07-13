"""MCP stdio protocol bridge for Trae compatibility.

Trae's MCP client uses Content-Length framed JSON-RPC (LSP style), while
paper-search-mcp (built on mcp-python-sdk >=1.6) uses newline-delimited
JSON-RPC. This bridge translates between the two formats so the server can
be used from Trae without modifying the installed package.
"""
import json
import subprocess
import sys
import threading


def _read_content_length_frame(stream):
    """Read one Content-Length framed JSON-RPC message from stream."""
    header = b""
    while True:
        byte = stream.read(1)
        if not byte:
            return None
        header += byte
        if header.endswith(b"\r\n\r\n"):
            break

    try:
        length_line = header.decode("utf-8", errors="replace").split("\r\n")[0]
        length = int(length_line.split(":", 1)[1].strip())
    except Exception as exc:
        raise ValueError(f"Invalid Content-Length header: {header!r}") from exc

    body = stream.read(length)
    if len(body) < length:
        return None
    return body.decode("utf-8", errors="replace")


def _write_content_length_frame(stream, message):
    """Write one Content-Length framed JSON-RPC message to stream."""
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    frame = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8") + body
    stream.write(frame)
    stream.flush()


def _read_ndjson_line(stream):
    """Read one newline-delimited JSON-RPC message from stream."""
    line = stream.readline()
    if not line:
        return None
    return line.decode("utf-8", errors="replace")


def _write_ndjson_line(stream, message):
    """Write one newline-delimited JSON-RPC message to stream."""
    line = json.dumps(message, ensure_ascii=False) + "\n"
    stream.write(line.encode("utf-8"))
    stream.flush()


def _forward_stderr(server_process):
    """Forward server stderr to our stderr for debugging."""
    try:
        for line in server_process.stderr:
            sys.stderr.write(line.decode("utf-8", errors="replace"))
            sys.stderr.flush()
    except Exception:
        pass


def _forward_client_to_server(server_stdin):
    """Translate Trae's Content-Length frames to server's NDJSON."""
    try:
        while True:
            body = _read_content_length_frame(sys.stdin.buffer)
            if body is None:
                break
            message = json.loads(body)
            _write_ndjson_line(server_stdin, message)
    except Exception as exc:
        sys.stderr.write(f"[bridge] client->server error: {exc}\n")
        sys.stderr.flush()


def _forward_server_to_client(server_stdout):
    """Translate server's NDJSON to Trae's Content-Length frames."""
    try:
        while True:
            line = _read_ndjson_line(server_stdout)
            if line is None:
                break
            message = json.loads(line)
            _write_content_length_frame(sys.stdout.buffer, message)
    except Exception as exc:
        sys.stderr.write(f"[bridge] server->client error: {exc}\n")
        sys.stderr.flush()


def main():
    server = subprocess.Popen(
        [sys.executable, "-m", "paper_search_mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    threading.Thread(target=_forward_stderr, args=(server,), daemon=True).start()

    t_in = threading.Thread(target=_forward_client_to_server, args=(server.stdin,))
    t_out = threading.Thread(target=_forward_server_to_client, args=(server.stdout,))
    t_in.start()
    t_out.start()
    t_in.join()
    t_out.join()

    try:
        server.kill()
    except Exception:
        pass


if __name__ == "__main__":
    main()
