"""
IT Helpdesk MCP Server
Exposes two tools over MCP (streamable HTTP):
    - create_support_ticket : create a new IT support ticket
    - check_ticket_status   : get the status of an existing ticket
Tickets are stored in an in-memory Python dictionary.

Run (keep this terminal open):
    python mcp_server.py
Server URL: http://localhost:8000/mcp
"""

import json
from datetime import datetime
from pathlib import Path
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("IT Helpdesk MCP Server")

# In-memory ticket store, loaded from mcp_tool_data.json
BASE_DIR = Path(__file__).parent


def load_tickets():
    candidates = [
        BASE_DIR / "mcp_tool_data.json",
        BASE_DIR.parent.parent / "Docs" / "Assessments" / "Day-12" / "mcp_tool_data.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {t["ticket_id"]: t for t in data}
    return {}


tickets = load_tickets()
VALID_PRIORITIES = ("Low", "Medium", "High")


def next_ticket_id():
    """Highest existing IT-<number> + 1, so IDs never repeat."""
    numbers = [int(t.split("-")[1]) for t in tickets if t.startswith("IT-") and t.split("-")[1].isdigit()]
    return f"IT-{max(numbers, default=100) + 1}"


@mcp.tool()
def create_support_ticket(employee: str, issue: str, description: str = "", priority: str = "Medium") -> str:
    """Create a new IT support ticket.

    Args:
        employee: Name of the employee raising the ticket.
        issue: Short title of the IT problem, e.g. "Laptop running slowly".
        description: More detail about the problem.
        priority: Low, Medium or High (default Medium).
    """
    priority = priority.strip().capitalize()
    if priority not in VALID_PRIORITIES:
        return json.dumps({"error": f"Invalid priority '{priority}'. Use Low, Medium or High."})
    if not employee.strip() or not issue.strip():
        return json.dumps({"error": "Employee name and issue are required."})

    ticket_id = next_ticket_id()
    tickets[ticket_id] = {
        "ticket_id": ticket_id,
        "employee": employee.strip(),
        "issue": issue.strip(),
        "description": description.strip() or issue.strip(),
        "priority": priority,
        "status": "Open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return json.dumps({"message": "Ticket created successfully", **tickets[ticket_id]})


@mcp.tool()
def check_ticket_status(ticket_id: str) -> str:
    """Check the status of an existing IT support ticket.

    Args:
        ticket_id: The ticket ID, for example IT-101.
    """
    ticket = tickets.get(ticket_id.strip().upper())
    if ticket is None:
        return json.dumps({"error": f"Ticket {ticket_id} not found"})
    return json.dumps(ticket)


if __name__ == "__main__":
    print("IT Helpdesk MCP Server running at http://localhost:8000/mcp")
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
