"""MCP mock tools router for testing tool-use capable models."""

import ast
import operator
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from mlx_manager.dependencies import get_current_user
from mlx_manager.models import User

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


# Safe arithmetic operators for the calculator
SAFE_OPERATORS: dict[type[ast.operator | ast.unaryop], Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval_expr(node: ast.AST) -> float:
    """Safely evaluate an arithmetic expression AST node.

    Only allows numeric literals and basic arithmetic operators.
    Raises ValueError for any unsupported expression elements.
    """
    if isinstance(node, ast.Expression):
        return safe_eval_expr(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    elif isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        left = safe_eval_expr(node.left)
        right = safe_eval_expr(node.right)
        op_func = SAFE_OPERATORS[type(node.op)]
        return float(op_func(left, right))  # type: ignore[arg-type]
    elif isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        op_func = SAFE_OPERATORS[type(node.op)]
        return float(op_func(safe_eval_expr(node.operand)))  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def execute_get_weather(arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute the get_weather mock tool.

    Returns deterministic mock weather data based on location hash.
    """
    location = arguments.get("location", "")
    unit = arguments.get("unit", "celsius")

    if not location:
        return {"error": "Location is required"}

    # Generate deterministic mock data based on location
    location_hash = hash(location.lower())

    # Temperature range: 10-35°C or 50-95°F
    if unit == "fahrenheit":
        temp_min, temp_max = 50, 95
    else:
        temp_min, temp_max = 10, 35

    temperature = temp_min + (abs(location_hash) % (temp_max - temp_min))

    # Conditions based on hash
    conditions = ["sunny", "cloudy", "partly cloudy", "rainy", "windy"]
    condition = conditions[abs(location_hash) % len(conditions)]

    # Humidity 30-90%
    humidity = 30 + (abs(location_hash >> 8) % 61)

    return {
        "location": location,
        "temperature": temperature,
        "unit": unit,
        "condition": condition,
        "humidity": humidity,
    }


def execute_calculate(arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute the calculate mock tool using safe AST evaluation.

    Only allows numeric literals and basic arithmetic operators (+, -, *, /, **, unary -).
    Rejects any code injection attempts.
    """
    expression = arguments.get("expression", "")

    if not expression:
        return {"error": "Expression is required"}

    try:
        tree = ast.parse(expression, mode="eval")
        result = safe_eval_expr(tree)
        return {"expression": expression, "result": result}
    except (SyntaxError, ValueError) as e:
        return {"error": f"Calculation failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# Tool execution registry
TOOL_EXECUTORS = {
    "get_weather": execute_get_weather,
    "calculate": execute_calculate,
}


class ToolExecuteRequest(BaseModel):
    """Request model for tool execution."""

    name: str
    arguments: dict[str, Any]


@router.get("/tools")
async def list_tools(
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[dict[str, Any]]:
    """List available mock tools in OpenAI function-calling format.

    Returns tool definitions that can be used with models that support tool calling.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and country, e.g. Tokyo, Japan",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Defaults to celsius.",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": (
                    "Perform a mathematical calculation. "
                    "Supports basic arithmetic: +, -, *, /, ** (power), and parentheses."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": (
                                "The mathematical expression to evaluate, e.g. '2 + 3 * 4'"
                            ),
                        },
                    },
                    "required": ["expression"],
                },
            },
        },
    ]
    return tools


@router.post("/execute")
async def execute_tool(
    current_user: Annotated[User, Depends(get_current_user)],
    request: ToolExecuteRequest,
) -> dict[str, Any]:
    """Execute a tool call and return the result.

    The tool name and arguments should match the definitions from /api/mcp/tools.
    Returns the tool execution result or an error if the tool fails.
    """
    tool_name = request.name
    executor = TOOL_EXECUTORS.get(tool_name)

    if not executor:
        return {"error": f"Unknown tool: {tool_name}"}

    # Execute the tool and return result
    result = executor(request.arguments)
    return result
