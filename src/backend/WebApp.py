import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Tuple

try:
    from backend import config
    from backend.PipelineService import pipeline_service
except ImportError:
    import config  # type: ignore
    from PipelineService import pipeline_service  # type: ignore


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


class QueryHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Tuple[object, ...]) -> None:
        # Keep terminal clean while serving browser requests.
        return

    def do_OPTIONS(self) -> None:
        _json_response(self, 200, {"ok": True})

    def do_GET(self) -> None:
        if self.path == "/api/nodes":
            try:
                with open(config.CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
                    nodes = json.load(f)
            except FileNotFoundError:
                _json_response(self, 404, {"error": "chunks metadata not found"})
                return
            except json.JSONDecodeError:
                _json_response(self, 500, {"error": "chunks metadata is invalid"})
                return
            except Exception as exc:
                _json_response(self, 500, {"error": f"failed to load nodes: {exc}"})
                return

            if not isinstance(nodes, list):
                _json_response(self, 500, {"error": "chunks metadata format is invalid"})
                return

            _json_response(self, 200, {"nodes": nodes})
            return

        if self.path == "/api/edges":
            try:
                with open(config.GRAPH_EDGES_PATH, "r", encoding="utf-8") as f:
                    edge_data = json.load(f)
            except FileNotFoundError:
                _json_response(self, 404, {"error": "graph edges not found"})
                return
            except json.JSONDecodeError:
                _json_response(self, 500, {"error": "graph edges is invalid"})
                return
            except Exception as exc:
                _json_response(self, 500, {"error": f"failed to load edges: {exc}"})
                return

            if not isinstance(edge_data, dict):
                _json_response(self, 500, {"error": "graph edges format is invalid"})
                return

            edges = edge_data.get("edges", [])
            if not isinstance(edges, list):
                _json_response(self, 500, {"error": "graph edges list is invalid"})
                return

            _json_response(self, 200, {"edges": edges})
            return

        if (self.path == "/api/expert-paths"):
            try:
                with open(config.EXPERT_PATHS_PATH, "r", encoding="utf-8") as f:
                    expert_data = json.load(f)
            except FileNotFoundError:
                _json_response(self, 404, {"error": "expert paths not found"})
                return
            except Exception as exc:
                _json_response(self, 500, {"error": f"failed to load expert paths: {exc}"})
                return

            _json_response(self, 200, expert_data)
            return

        _json_response(self, 404, {"error": "Not Found"})

    def do_POST(self) -> None:
        if self.path != "/api/query":
            _json_response(self, 404, {"error": "Not Found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            _json_response(self, 400, {"error": "Invalid JSON body"})
            return

        query = str(payload.get("query", "")).strip()
        answer_language = payload.get("answer_language")

        if not query:
            _json_response(self, 400, {"error": "query is required"})
            return

        try:
            output = pipeline_service.answer_query(query=query, answer_language=answer_language)
        except Exception as exc:
            _json_response(self, 500, {"error": f"Pipeline failed: {exc}"})
            return

        _json_response(
            self,
            200,
            {
                "query": query,
                "llm_response": output.get("llm_response", ""),
                "top3_nodes": output.get("top3_nodes", []),
                "directly_related_nodes": output.get("directly_related_nodes", []),
                "query_plan": output.get("query_plan", {}),
            },
        )


def run_server() -> None:
    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT", "8001"))
    try:
        server = ThreadingHTTPServer((host, port), QueryHandler)
    except OSError as exc:
        if getattr(exc, "errno", None) == 98:
            print(
                f"Port {port} is already in use. Stop the existing backend process "
                "or run with BACKEND_PORT=<new_port>."
            )
            raise SystemExit(1)
        raise
    print(f"Backend API listening at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
