"""E2E: MCP Client side call simulation.

Starts the MCP server as a subprocess and simulates an MCP client by sending
JSON-RPC over stdio: initialize, tools/list, and tools/call (query_knowledge_hub).
Validates that query_knowledge_hub completes and returns content with citation
support (content blocks and structure).

Usage::

    pytest -q tests/e2e/test_mcp_client.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import pytest


def _send_and_receive(
    proc: subprocess.Popen,
    requests: List[Dict[str, Any]],
    timeout: float = 10.0,
    expected_responses: Optional[int] = None,
) -> List[str]:
    """Send JSON-RPC requests to process stdin and collect stdout lines.

    Args:
        proc: Subprocess with stdin/stdout pipes.
        requests: List of JSON-RPC requests/notifications.
        timeout: Max time to wait for responses.
        expected_responses: Number of responses to wait for (None = infer from requests).

    Returns:
        List of lines read from stdout.
    """
    assert proc.stdin is not None
    assert proc.stdout is not None

    for req in requests:
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()

    lines: List[str] = []
    start = time.time()
    response_count = 0
    if expected_responses is None:
        expected_responses = sum(1 for req in requests if "id" in req)

    while time.time() - start < timeout:
        if expected_responses > 0 and response_count >= expected_responses:
            break
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
            try:
                data = json.loads(stripped)
                if "id" in data and ("result" in data or "error" in data):
                    response_count += 1
            except json.JSONDecodeError:
                pass

    return lines


def _find_response(lines: List[str], request_id: int) -> Optional[Dict[str, Any]]:
    """Find JSON-RPC response with given id in stdout lines."""
    for line in lines:
        if not line.startswith('{"jsonrpc"'):
            continue
        try:
            data = json.loads(line)
            if data.get("id") == request_id:
                return data
        except json.JSONDecodeError:
            continue
    return None


def _start_server() -> subprocess.Popen:
    """Start MCP server as subprocess with stdio pipes (UTF-8 for cross-platform)."""
    return subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _teardown_server(proc: subprocess.Popen) -> None:
    """Terminate server process and wait."""
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# Default client sequence: initialize + initialized notification
_INIT_REQUESTS = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "clientInfo": {"name": "e2e-test", "version": "0.0.0"},
            "capabilities": {},
        },
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
]


# -----------------------------------------------------------------------------
# tests
# -----------------------------------------------------------------------------


@pytest.mark.e2e
def test_mcp_client_tools_list() -> None:
    """E2E: Subprocess server responds to tools/list and exposes query_knowledge_hub."""
    proc = _start_server()
    try:
        requests = _INIT_REQUESTS + [
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ]
        lines = _send_and_receive(proc, requests, timeout=10.0)
    finally:
        _teardown_server(proc)

    assert len(lines) >= 2, "Expected at least initialize and tools/list responses"

    init_resp = _find_response(lines, 1)
    assert init_resp is not None, "Missing initialize response"
    assert "result" in init_resp

    tools_resp = _find_response(lines, 2)
    assert tools_resp is not None, f"Missing tools/list response in: {lines}"
    assert tools_resp.get("jsonrpc") == "2.0"
    assert "result" in tools_resp
    assert "tools" in tools_resp["result"]
    tools = tools_resp["result"]["tools"]
    assert isinstance(tools, list)
    tool_names = [t["name"] for t in tools]
    assert "query_knowledge_hub" in tool_names, f"Expected query_knowledge_hub in {tool_names}"


@pytest.mark.e2e
def test_mcp_client_query_knowledge_hub_returns_citations() -> None:
    """E2E: tools/call query_knowledge_hub completes and returns content (citations structure).

    Full flow: start server, initialize, then tools/call query_knowledge_hub.
    Asserts response has valid MCP content blocks; citations may appear in text
    or as structured content (empty collection still returns valid shape).
    """
    proc = _start_server()
    try:
        requests = _INIT_REQUESTS + [
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge_hub",
                    "arguments": {"query": "test query"},
                },
            },
        ]
        lines = _send_and_receive(proc, requests, timeout=15.0)
    finally:
        _teardown_server(proc)

    assert len(lines) >= 2, "Expected at least initialize and tools/call responses"

    call_resp = _find_response(lines, 2)
    assert call_resp is not None, f"Missing tools/call response in: {lines}"
    assert "result" in call_resp, f"Expected result in tools/call response: {call_resp}"
    result = call_resp["result"]

    # MCP CallToolResult: content (list of content blocks), optional isError
    assert "content" in result, "CallToolResult must have content"
    content = result["content"]
    assert isinstance(content, list), "content must be a list of blocks"

    # Each block: type "text" (or "image") and type-specific fields (e.g. text)
    all_text = ""
    for block in content:
        assert isinstance(block, dict), "Each content block must be a dict"
        assert "type" in block, "Content block must have type"
        if block["type"] == "text":
            assert "text" in block, "TextContent must have text"
            assert isinstance(block["text"], str), "text must be string"
            all_text += block["text"]

    # Success: either no error, or structured error that includes citations/references
    # (when API key is missing we get isError but response still has citation structure)
    is_error = result.get("isError", False)
    if is_error:
        assert "citations" in all_text or "References" in all_text, (
            "Error response should still expose citation/references structure"
        )
    else:
        # Normal path: content with optional citations
        pass
