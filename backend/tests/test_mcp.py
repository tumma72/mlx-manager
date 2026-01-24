"""Tests for MCP mock tools endpoints."""

import pytest


@pytest.mark.asyncio
async def test_list_tools(auth_client):
    """Test listing available mock tools."""
    response = await auth_client.get("/api/mcp/tools")
    assert response.status_code == 200

    tools = response.json()
    assert len(tools) == 2

    # Check get_weather tool
    weather_tool = next(t for t in tools if t["function"]["name"] == "get_weather")
    assert weather_tool["type"] == "function"
    assert "location" in weather_tool["function"]["parameters"]["properties"]
    assert "unit" in weather_tool["function"]["parameters"]["properties"]
    assert weather_tool["function"]["parameters"]["required"] == ["location"]

    # Check calculate tool
    calc_tool = next(t for t in tools if t["function"]["name"] == "calculate")
    assert calc_tool["type"] == "function"
    assert "expression" in calc_tool["function"]["parameters"]["properties"]
    assert calc_tool["function"]["parameters"]["required"] == ["expression"]


@pytest.mark.asyncio
async def test_execute_weather(auth_client):
    """Test executing get_weather tool."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["location"] == "Tokyo"
    assert data["unit"] == "celsius"
    assert "temperature" in data
    assert "condition" in data
    assert "humidity" in data
    assert isinstance(data["temperature"], int)
    assert isinstance(data["humidity"], int)


@pytest.mark.asyncio
async def test_execute_weather_default_unit(auth_client):
    """Test get_weather defaults to celsius when unit not specified."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "get_weather", "arguments": {"location": "Paris"}},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["location"] == "Paris"
    assert data["unit"] == "celsius"


@pytest.mark.asyncio
async def test_execute_weather_fahrenheit(auth_client):
    """Test get_weather with fahrenheit unit."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "get_weather", "arguments": {"location": "New York", "unit": "fahrenheit"}},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["location"] == "New York"
    assert data["unit"] == "fahrenheit"
    # Fahrenheit should be in 50-95 range
    assert 50 <= data["temperature"] <= 95


@pytest.mark.asyncio
async def test_execute_weather_deterministic(auth_client):
    """Test that weather results are deterministic for same location."""
    # Call twice with same location
    response1 = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "get_weather", "arguments": {"location": "London"}},
    )
    response2 = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "get_weather", "arguments": {"location": "London"}},
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    # Results should be identical
    assert data1 == data2


@pytest.mark.asyncio
async def test_execute_calculate(auth_client):
    """Test calculating basic arithmetic expression."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "calculate", "arguments": {"expression": "2 + 3 * 4"}},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["expression"] == "2 + 3 * 4"
    assert data["result"] == 14.0


@pytest.mark.asyncio
async def test_execute_calculate_division(auth_client):
    """Test calculate with division returning float."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "calculate", "arguments": {"expression": "10 / 3"}},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["expression"] == "10 / 3"
    assert abs(data["result"] - 3.333333) < 0.00001


@pytest.mark.asyncio
async def test_execute_calculate_power(auth_client):
    """Test calculate with power operator."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "calculate", "arguments": {"expression": "2 ** 8"}},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["result"] == 256.0


@pytest.mark.asyncio
async def test_execute_calculate_parentheses(auth_client):
    """Test calculate with parentheses."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "calculate", "arguments": {"expression": "(2 + 3) * 4"}},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["result"] == 20.0


@pytest.mark.asyncio
async def test_execute_calculate_negative(auth_client):
    """Test calculate with unary minus."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "calculate", "arguments": {"expression": "-5 + 10"}},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["result"] == 5.0


@pytest.mark.asyncio
async def test_execute_calculate_invalid_import(auth_client):
    """Test calculate rejects code injection attempts (import)."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "calculate", "arguments": {"expression": "import os"}},
    )
    assert response.status_code == 200

    data = response.json()
    assert "error" in data
    assert "Calculation failed" in data["error"]


@pytest.mark.asyncio
async def test_execute_calculate_invalid_function_call(auth_client):
    """Test calculate rejects function calls."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "calculate", "arguments": {"expression": "__import__('os')"}},
    )
    assert response.status_code == 200

    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_execute_calculate_invalid_syntax(auth_client):
    """Test calculate handles invalid syntax gracefully."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "calculate", "arguments": {"expression": "2 + + 3"}},
    )
    assert response.status_code == 200

    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_execute_calculate_empty_expression(auth_client):
    """Test calculate handles empty expression."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "calculate", "arguments": {"expression": ""}},
    )
    assert response.status_code == 200

    data = response.json()
    assert "error" in data
    assert "required" in data["error"].lower()


@pytest.mark.asyncio
async def test_execute_unknown_tool(auth_client):
    """Test executing unknown tool returns error."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "nonexistent_tool", "arguments": {}},
    )
    assert response.status_code == 200

    data = response.json()
    assert "error" in data
    assert "Unknown tool" in data["error"]


@pytest.mark.asyncio
async def test_tools_require_auth(client):
    """Test that unauthenticated request to tools endpoint fails."""
    response = await client.get("/api/mcp/tools")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_execute_requires_auth(client):
    """Test that unauthenticated request to execute endpoint fails."""
    response = await client.post(
        "/api/mcp/execute",
        json={"name": "calculate", "arguments": {"expression": "1+1"}},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_execute_weather_missing_location(auth_client):
    """Test get_weather with missing location returns error."""
    response = await auth_client.post(
        "/api/mcp/execute",
        json={"name": "get_weather", "arguments": {}},
    )
    assert response.status_code == 200

    data = response.json()
    assert "error" in data
    assert "required" in data["error"].lower()
