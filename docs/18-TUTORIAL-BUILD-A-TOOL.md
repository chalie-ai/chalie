# Tutorial: Build a Custom Tool for Chalie

Welcome! In this tutorial, you'll learn how to extend Chalie's capabilities by building your own custom tool. By the end, you'll have created a working tool that can be invoked during conversations with Chalie.

## Prerequisites

- You've completed [Tutorial: Your First 10 Minutes](./17-TUTORIAL-FIRST-10-MINUTES.md)
- Basic understanding of Python and JSON
- Familiarity with the [Tools System architecture](./09-TOOLS.md)

---

## What You'll Build

We'll create a **`weather_lookup`** tool that:
- Accepts a city name as input
- Returns weather information in both text and HTML card format
- Demonstrates all four key components of a Chalie tool

> **Quick Fact**: Tools can run either as **sandboxed** (Docker container) or **trusted** (Python subprocess). This tutorial covers trusted tools for simplicity, but the concepts apply to both.

---

## Tool Anatomy: Four Key Components

Every Chalie tool consists of four essential parts:

| Component | Purpose | File |
|------|--|-|
| **Manifest** | Declares what your tool does and how it's triggered | `manifest.json` |
| **Handler** | Business logic that processes the request | `handler.py` |
| **Runner** | Entry point that receives framework input and calls handler | `runner.py` |
| **Tests** | Validates your tool works correctly | `test_*.py` (optional but recommended) |

Let's build each one.

---

## Step 1: Create the Tool Directory

Create a new directory for your tool in `backend/tools/`:

```bash
mkdir -p backend/tools/weather_lookup
cd backend/tools/weather_lookup
```

Your final structure will look like this:

```
weather_lookup/
├── manifest.json    # Tool declaration
├── handler.py       # Business logic
├── runner.py        # Entry point
└── test_weather.py  # Tests (optional)
```

---

## Step 2: Write the Manifest (`manifest.json`)

The manifest is a JSON file that declares your tool's identity, parameters, and behavior. It's how Chalie understands what your tool does and when to use it.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique identifier (lowercase alphanumeric with underscores/hyphens) |
| `description` | string | Human-readable description for Chalie's semantic matching |
| `trigger` | object | How the tool is invoked (`on_demand`, `cron`, or `webhook`) |
| `parameters` | object | Input schema that Chalie will extract from user intent |
| `returns` | object | Output schema your tool produces |

### Complete Example

Create `manifest.json`:

```json
{
  "name": "weather_lookup",
  "description": "Get current weather information for a city. Returns temperature, conditions, and humidity in a formatted card.",
  "version": "1.0.0",
  "category": "search",
  
  "trigger": {
    "type": "on_demand"
  },

  "parameters": {
    "city": {
      "type": "string",
      "description": "The name of the city to get weather for (e.g., 'London', 'Tokyo')",
      "required": true,
      "default": null
    }
  },

  "returns": {
    "text": {
      "type": "string",
      "description": "Human-readable weather summary"
    },
    "html": {
      "type": "string",
      "description": "HTML card with weather information and styling"
    }
  },

  "output": {
    "synthesize": true,
    "mode": "immediate"
  },

  "card": {
    "title": "Weather Report",
    "icon": "🌤️"
  },

  "documentation": "https://github.com/your-org/chalie-tools/weather_lookup#readme",

  "config_schema": [
    {
      "key": "api_key",
      "type": "string",
      "description": "Weather API key (optional, tool uses mock data if not provided)",
      "required": false,
      "sensitive": true
    }
  ]
}
```

### Field Deep Dive

#### `trigger.type` Options

| Value | When It Runs | Use Case |
|-------|--------------|----------|
| `on_demand` | Only when Chalie decides to invoke it | Most tools (search, calculation, etc.) |
| `cron` | On a schedule (e.g., `"0 * * * *"`) | Periodic tasks, data syncs |
| `webhook` | When external HTTP request arrives | Integrations with external services |

#### `output.mode` Options

| Value | Behavior | Use Case |
|-------|----------|----------|
| `immediate` | Result shown right away | Most tools |
| `deferred` | Card rendered after conversation turn | Long-running operations, rich reports |

#### `config_schema`

Defines configuration keys users can set via the UI or API. Set `"sensitive": true` for secrets (they'll be masked in responses).

> **Decision Guide**: Should your tool use `synthesize: true` or `false`?
> - **Use `true`** if you want Chalie to rewrite your output in its conversational voice
> - **Use `false`** if your text should appear verbatim (e.g., code snippets, exact quotes)

---

## Step 3: Write the Handler (`handler.py`)

The handler contains your tool's business logic. It receives a structured payload and returns results.

### The Payload Structure

Chalie sends this to your tool:

```python
{
    "params": {           # Extracted from user intent per manifest.parameters
        "city": "London"
    },
    "settings": {         # Config values from database (per config_schema)
        "api_key": "abc123..."  # Only if user configured it
    },
    "telemetry": {        # Client context (always present, fields may be null)
        "lat": 51.5074,
        "lon": -0.1278,
        "city": "London",
        "country": "UK",
        "time": "2026-03-07T12:00:00Z",
        "locale": "en-GB"
    }
}
```

### Complete Handler Example

Create `handler.py`:

```python
"""
Weather Lookup Tool Handler

Processes weather lookup requests and returns formatted results.
This is a mock implementation — replace with real API calls in production.
"""

import json
from datetime import datetime


def handle(payload: dict) -> dict:
    """
    Process a weather lookup request.

    Args:
        payload: Dict containing 'params', 'settings', and 'telemetry' keys.
            - params.city (str): City name to look up
            - settings.api_key (str, optional): Weather API key if configured
            - telemetry: Client context with location/time info

    Returns:
        dict with keys:
            - text (str): Human-readable weather summary
            - html (str): HTML card fragment for UI display
            - error (str, optional): Error message if something failed

    Example:
        >>> payload = {"params": {"city": "London"}, "settings": {}, "telemetry": {}}
        >>> result = handle(payload)
        >>> "text" in result and "html" in result
        True
    """
    params = payload.get("params", {})
    settings = payload.get("settings", {})
    telemetry = payload.get("telemetry", {})

    # Extract city name (required per manifest)
    city = params.get("city")
    if not city:
        return {
            "error": "City parameter is required"
        }

    try:
        # In production, replace this with a real API call using settings.api_key
        weather_data = _get_weather_for_city(city)
        
        # Generate text summary
        text = _generate_text_summary(weather_data)
        
        # Generate HTML card
        html = _generate_html_card(weather_data)

        return {
            "text": text,
            "html": html
        }

    except Exception as e:
        return {
            "error": f"Failed to fetch weather data: {str(e)}"
        }


def _get_weather_for_city(city: str) -> dict:
    """
    Fetch weather data for a city.

    This is a MOCK implementation that returns deterministic fake data.
    Replace with real API calls (OpenWeatherMap, WeatherAPI, etc.).

    Args:
        city: Name of the city to look up

    Returns:
        dict with temperature, conditions, humidity, and timestamp
    """
    # Mock data generator — replace with real API in production
    import hashlib
    
    # Create deterministic "random" values based on city name
    hash_val = int(hashlib.md5(city.lower().encode()).hexdigest()[:8], 16)
    
    temperature_celsius = 10 + (hash_val % 25)  # 10-34°C
    humidity = 40 + (hash_val % 50)             # 40-89%
    
    conditions_map = ["Sunny", "Cloudy", "Partly Cloudy", "Rainy", "Stormy"]
    condition = conditions_map[hash_val % len(conditions_map)]

    return {
        "city": city,
        "temperature_celsius": temperature_celsius,
        "condition": condition,
        "humidity_percent": humidity,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def _generate_text_summary(weather: dict) -> str:
    """
    Generate a human-readable text summary of weather data.

    Args:
        weather: Dict from _get_weather_for_city()

    Returns:
        Formatted string with weather information
    """
    temp_f = int(weather["temperature_celsius"] * 9/5 + 32)
    
    return (
        f"Weather in {weather['city']}: {weather['condition']}.\n"
        f"Temperature: {weather['temperature_celsius']}°C ({temp_f}°F).\n"
        f"Humidity: {weather['humidity_percent']}%."
    )


def _generate_html_card(weather: dict) -> str:
    """
    Generate an HTML card fragment for weather display.

    Follows Chalie's HTML rules:
    - Inline CSS only (no <style> blocks or external stylesheets)
    - No JavaScript (no <script>, no event handlers, no javascript: URIs)
    - Fragment only (no <html>, <head>, <body>)
    - No dangerous tags (<iframe>, <form>, <input>, etc.)

    Args:
        weather: Dict from _get_weather_for_city()

    Returns:
        HTML string fragment for UI card display
    """
    temp_f = int(weather["temperature_celsius"] * 9/5 + 32)
    
    # Choose icon based on condition
    icon_map = {
        "Sunny": "☀️",
        "Cloudy": "☁️",
        "Partly Cloudy": "⛅",
        "Rainy": "🌧️",
        "Stormy": "⚡"
    }
    icon = icon_map.get(weather["condition"], "🌤️")

    html = f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 12px; padding: 20px; color: white; max-width: 300px;">
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span style="font-size: 48px; margin-right: 15px;">{icon}</span>
        <div>
            <h2 style="margin: 0; font-size: 20px;">{weather['city']}</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">{weather['condition']}</p>
        </div>
    </div>
    <div style="display: flex; justify-content: space-around; text-align: center;">
        <div>
            <div style="font-size: 32px; font-weight: bold;">
                {weather['temperature_celsius']}°C
            </div>
            <div style="opacity: 0.8; font-size: 14px;">{temp_f}°F</div>
        </div>
        <div>
            <div style="font-size: 24px;">💧</div>
            <div style="font-size: 16px;">{weather['humidity_percent']}%</div>
            <div style="opacity: 0.8; font-size: 12px;">Humidity</div>
        </div>
    </div>
</div>
"""
    return html.strip()


# For testing purposes, allow direct invocation
if __name__ == "__main__":
    test_payload = {
        "params": {"city": "London"},
        "settings": {},
        "telemetry": {}
    }
    
    result = handle(test_payload)
    print(json.dumps(result, indent=2))
```

### Handler Best Practices

1. **Always validate inputs** — Check required parameters exist before processing
2. **Return errors in the `error` key** — Don't raise exceptions; return them as structured errors
3. **Keep it pure** — Your handler should be a function, not a class (makes testing easier)
4. **Separate concerns** — Use helper functions for data fetching, text generation, HTML rendering

---

## Step 4: Write the Runner (`runner.py`)

The runner is your tool's entry point. It receives base64-encoded JSON from Chalie, decodes it, calls your handler, and outputs JSON to stdout.

### The IPC Contract

Chalie communicates with tools via this protocol:
- **Input**: Base64-encoded JSON string passed as command-line argument
- **Output**: JSON object written to stdout (and only stdout)
- **Errors**: Non-zero exit code + error message on stderr

### Complete Runner Example

Create `runner.py`:

```python
#!/usr/bin/env python3
"""
Weather Lookup Tool Runner

Entry point for the weather_lookup tool. Receives base64-encoded JSON from Chalie,
decodes it, calls the handler, and outputs JSON to stdout.

Usage:
    python runner.py <base64_encoded_payload>

The payload structure (after decoding):
{
    "params": {"city": "..."},      # From manifest.parameters
    "settings": {...},              # Config from database
    "telemetry": {...}              # Client context
}

Output format:
{
    "text": "...",                  # Human-readable result (optional)
    "html": "...",                  # HTML card fragment (optional)
    "error": "..."                 # Error message if failed (optional)
}
"""

import sys
import base64
import json


def main():
    """
    Main entry point for the tool.

    Expects exactly one command-line argument: base64-encoded JSON payload.
    Writes result JSON to stdout, errors to stderr.
    
    Exit codes:
        0 — Success (even if handler returned an error in the response)
        1 — Fatal error (invalid input, encoding failure, etc.)
    """
    # Validate command-line arguments
    if len(sys.argv) != 2:
        print("Usage: runner.py <base64_encoded_payload>", file=sys.stderr)
        sys.exit(1)

    encoded_payload = sys.argv[1]

    try:
        # Decode the base64 payload
        decoded_bytes = base64.b64decode(encoded_payload)
        payload = json.loads(decoded_bytes.decode("utf-8"))
        
    except Exception as e:
        if isinstance(e, (ValueError, UnicodeDecodeError)):
            print(f"Invalid base64 encoding: {e}", file=sys.stderr)
        else:
            print(f"Invalid JSON payload: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Import and call the handler
        from handler import handle
        
        result = handle(payload)
        
        # Output result to stdout (Chalie reads this)
        print(json.dumps(result))
        
    except Exception as e:
        # Handler raised an unexpected exception — output error JSON
        error_result = {"error": f"Tool execution failed: {str(e)}"}
        print(json.dumps(error_result))


if __name__ == "__main__":
    main()
```

### Runner Anatomy Explained

| Section | Purpose |
|---------|---------|
| Argument validation | Ensures exactly one base64 argument is provided |
| Base64 decode + JSON parse | Converts framework input to Python dict |
| Handler import + call | Executes your business logic |
| JSON output to stdout | Returns result in the format Chalie expects |

> **Quick Fact**: The runner always exits with code 0 unless there's a fatal error (invalid encoding, missing argument). If your handler returns an `error` key, that's still considered success — Chalie handles the error gracefully.

---

## Step 5: Test Your Tool

Testing ensures your tool works correctly before deploying it to production.

### Manual Testing

Test your runner directly from the command line:

```bash
cd backend/tools/weather_lookup

# Create a test payload and run it
python3 -c "
import base64, json
payload = {
    'params': {'city': 'London'},
    'settings': {},
    'telemetry': {}
}
print(base64.b64encode(json.dumps(payload).encode()).decode())
" | xargs python3 runner.py
```

Expected output:
```json
{
  "text": "Weather in London: Sunny.\nTemperature: 20°C (68°F).\nHumidity: 65%.",
  "html": "<div style=\"...\">...</div>"
}
```

### Automated Testing with `test_weather.py`

Create a test file to validate your handler logic:

```python
"""
Tests for weather_lookup tool handler.

Run with: python -m pytest test_weather.py -v
Or:       python test_weather.py (if using assert-based tests)
"""

import json
from handler import handle, _get_weather_for_city


def test_handle_with_valid_city():
    """Test that a valid city returns weather data."""
    payload = {
        "params": {"city": "London"},
        "settings": {},
        "telemetry": {}
    }
    
    result = handle(payload)
    
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert "text" in result, "Result should contain 'text' key"
    assert "html" in result, "Result should contain 'html' key"
    assert "London" in result["text"], "Text should mention the city name"


def test_handle_with_missing_city():
    """Test that missing city parameter returns an error."""
    payload = {
        "params": {},
        "settings": {},
        "telemetry": {}
    }
    
    result = handle(payload)
    
    assert "error" in result, "Result should contain 'error' key when city is missing"
    assert "required" in result["error"].lower(), "Error message should mention required parameter"


def test_handle_with_empty_city():
    """Test that empty city string returns an error."""
    payload = {
        "params": {"city": ""},
        "settings": {},
        "telemetry": {}
    }
    
    result = handle(payload)
    
    assert "error" in result, "Result should contain 'error' key for empty city"


def test_get_weather_for_city_returns_valid_structure():
    """Test that _get_weather_for_city returns expected fields."""
    weather = _get_weather_for_city("Tokyo")
    
    assert "city" in weather
    assert "temperature_celsius" in weather
    assert "condition" in weather
    assert "humidity_percent" in weather
    assert "timestamp" in weather
    
    # Validate ranges for mock data
    assert 10 <= weather["temperature_celsius"] <= 34, "Temperature should be between 10-34°C"
    assert 40 <= weather["humidity_percent"] <= 89, "Humidity should be between 40-89%"


def test_deterministic_weather():
    """Test that same city always returns same weather (deterministic mock)."""
    weather1 = _get_weather_for_city("Paris")
    weather2 = _get_weather_for_city("Paris")
    
    assert weather1["temperature_celsius"] == weather2["temperature_celsius"], \
        "Same city should return same temperature"
    assert weather1["condition"] == weather2["condition"], \
        "Same city should return same condition"


def test_html_contains_required_elements():
    """Test that generated HTML contains expected elements."""
    payload = {
        "params": {"city": "Berlin"},
        "settings": {},
        "telemetry": {}
    }
    
    result = handle(payload)
    html = result.get("html", "")
    
    assert "<div" in html, "HTML should contain div elements"
    assert "style=" in html, "HTML should have inline styles"
    assert "Berlin" in html, "HTML should mention the city name"


def test_text_contains_temperature():
    """Test that text output contains temperature information."""
    payload = {
        "params": {"city": "New York"},
        "settings": {},
        "telemetry": {}
    }
    
    result = handle(payload)
    text = result.get("text", "")
    
    assert "°C" in text or "°F" in text, "Text should contain temperature with units"


# Run tests if executed directly
if __name__ == "__main__":
    test_handle_with_valid_city()
    print("✓ test_handle_with_valid_city passed")
    
    test_handle_with_missing_city()
    print("✓ test_handle_with_missing_city passed")
    
    test_handle_with_empty_city()
    print("✓ test_handle_with_empty_city passed")
    
    test_get_weather_for_city_returns_valid_structure()
    print("✓ test_get_weather_for_city_returns_valid_structure passed")
    
    test_deterministic_weather()
    print("✓ test_deterministic_weather passed")
    
    test_html_contains_required_elements()
    print("✓ test_html_contains_required_elements passed")
    
    test_text_contains_temperature()
    print("✓ test_text_contains_temperature passed")
    
    print("\nAll tests passed! ✓")
```

### Running Tests

```bash
# Run all tests with pytest
python -m pytest test_weather.py -v

# Or run directly (for quick checks)
python test_weather.py
```

---

## Step 6: Register Your Tool

Once your tool is built and tested, you need to register it with Chalie. The exact method depends on your deployment setup:

### For Development

Add your tool path to the tools configuration in `backend/config/tools.json`:

```json
{
  "weather_lookup": {
    "path": "./tools/weather_lookup",
    "type": "trusted"
  }
}
```

### For Production (Docker)

Build a Docker image for your tool and register it via the Chalie API:

```bash
# Build the container
docker build -t chalie-tools/weather_lookup .

# Register via API (requires admin credentials)
curl -X POST http://localhost:8080/api/tools/register \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"name": "weather_lookup", "image": "chalie-tools/weather_lookup"}'
```

---

## Summary

You've now built a complete Chalie tool! Here's what you created:

| File | Purpose | Lines of Code |
|------|---------|---------------|
| `manifest.json` | Tool declaration and schema | ~50 |
| `handler.py` | Business logic with mock data | ~180 |
| `runner.py` | Entry point for IPC | ~60 |
| `test_weather.py` | Test suite | ~120 |

### Next Steps

1. **Replace mock data** — Connect to a real weather API (OpenWeatherMap, WeatherAPI)
2. **Add error handling** — Handle API rate limits, network failures gracefully
3. **Add caching** — Cache results for the same city within a time window
4. **Support more parameters** — Add units preference (Celsius/Fahrenheit), forecast days

### Resources

- [Tools System Architecture](./09-TOOLS.md) — Deep dive into how tools work
- [Testing Guide](./12-TESTING.md) — Best practices for testing Chalie components
- [Default Tools Reference](./14-DEFAULT-TOOLS.md) — Examples of built-in tools

---

## Appendix: Common Patterns

### Pattern 1: External API Integration

```python
def _fetch_from_api(city: str, api_key: str) -> dict:
    """Fetch real weather data from an external API."""
    import urllib.request
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        raise RuntimeError(f"API request failed: {e}")
```

### Pattern 2: Caching Results

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def _get_weather_cached(city: str, timestamp_bucket: int) -> dict:
    """Cache weather results for 5-minute buckets."""
    return _fetch_from_api(city)
```

### Pattern 3: Rich HTML Cards with Multiple Sections

```python
def _generate_html_card(weather: dict) -> str:
    """Generate a multi-section weather card."""
    sections = [
        _render_header_section(weather),
        _render_temperature_section(weather),
        _render_details_section(weather),
        _render_forecast_section(weather.get("forecast", []))
    ]
    
    return f'<div class="weather-card">{"".join(sections)}</div>'
```

---

*Happy building! 🚀*
