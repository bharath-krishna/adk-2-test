import os

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

linear_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://mcp.linear.app/mcp",
        headers={"Authorization": f"Bearer {os.environ.get('LINEAR_API_KEY', '')}"},
    ),
)
