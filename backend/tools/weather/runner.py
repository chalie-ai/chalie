import sys
import json
import base64
from html import escape as _esc
from handler import execute


def _generate_weather_html(data: dict, telemetry_location: str = "") -> str:
    """Generate an inline HTML weather card from result dict."""
    if not isinstance(data, dict):
        return ""

    location = _esc(telemetry_location or data.get("location", "Unknown"))
    condition = _esc(str(data.get("condition", "")))
    temp_c = data.get("temperature_c", "?")
    humidity = data.get("humidity_pct", "?")
    wind = data.get("wind_kmh", "?")
    wind_dir = _esc(str(data.get("wind_direction", "")))

    # Tomorrow's forecast section
    tmrw_condition = data.get("forecast_tomorrow_condition")
    tmrw_max = data.get("forecast_tomorrow_max_c")
    tmrw_min = data.get("forecast_tomorrow_min_c")
    tmrw_rain = data.get("forecast_tomorrow_precip_chance_pct")

    forecast_html = ""
    if tmrw_condition or tmrw_rain is not None:
        temp_range = f"{tmrw_min}–{tmrw_max}°C" if tmrw_min is not None and tmrw_max is not None else ""
        stat_parts = []
        if temp_range:
            stat_parts.append(
                f'<span style="display:flex;align-items:center;gap:5px">'
                f'<i class="fas fa-temperature-half" style="color:#8A5CFF;font-size:0.72rem"></i>'
                f'{temp_range}</span>'
            )
        if tmrw_rain is not None:
            stat_parts.append(
                f'<span style="display:flex;align-items:center;gap:5px">'
                f'<i class="fas fa-cloud-rain" style="color:#00F0FF;font-size:0.72rem"></i>'
                f'{tmrw_rain}%</span>'
            )
        tmrw_cond_html = (
            f'<div style="color:rgba(234,230,242,0.68);font-size:0.875rem;margin-bottom:6px">'
            f'{_esc(str(tmrw_condition))}</div>'
        ) if tmrw_condition else ""
        stats_html = (
            f'<div style="display:flex;gap:14px;font-size:0.8rem;color:rgba(234,230,242,0.52)">'
            f'{"".join(stat_parts)}</div>'
        ) if stat_parts else ""
        forecast_html = (
            f'<div style="margin-top:14px;padding-top:12px;'
            f'border-top:1px solid rgba(138,92,255,0.18)">'
            f'<div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.08em;color:rgba(234,230,242,0.38);margin-bottom:8px">'
            f'<i class="fas fa-calendar-day" style="margin-right:5px"></i>Tomorrow</div>'
            f'{tmrw_cond_html}{stats_html}</div>'
        )

    html = (
        f'<div style="'
        f'background:linear-gradient(135deg,rgba(138,92,255,0.11) 0%,rgba(10,12,22,0.97) 65%);'
        f'padding:18px 20px;color:#eae6f2;'
        f'position:relative;overflow:hidden">'
        # top edge highlight
        f'<div style="position:absolute;top:0;left:16px;right:16px;height:1px;'
        f'background:linear-gradient(90deg,transparent,rgba(138,92,255,0.55),rgba(0,240,255,0.25),transparent)">'
        f'</div>'
        # location row
        f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:13px">'
        f'<i class="fas fa-location-dot" style="color:#8A5CFF;font-size:0.78rem"></i>'
        f'<span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.07em;color:rgba(234,230,242,0.48)">{location}</span>'
        f'</div>'
        # temperature + condition
        f'<div style="display:flex;align-items:flex-end;gap:14px;margin-bottom:14px">'
        f'<div style="font-size:2.6rem;font-weight:300;line-height:1;'
        f'color:#eae6f2;letter-spacing:-1px">{temp_c}°C</div>'
        f'<div style="padding-bottom:5px;color:rgba(234,230,242,0.68);font-size:0.9rem">{condition}</div>'
        f'</div>'
        # stats row
        f'<div style="display:flex;gap:20px">'
        f'<div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;color:rgba(234,230,242,0.52)">'
        f'<i class="fas fa-droplet" style="color:#00F0FF;font-size:0.73rem;width:11px;text-align:center"></i>'
        f'<span>{humidity}%</span></div>'
        f'<div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;color:rgba(234,230,242,0.52)">'
        f'<i class="fas fa-wind" style="color:#00F0FF;font-size:0.73rem;width:11px;text-align:center"></i>'
        f'<span>{wind} km/h {wind_dir}</span></div>'
        f'</div>'
        f'{forecast_html}'
        f'</div>'
    )

    return html


payload = json.loads(base64.b64decode(sys.argv[1]))

# Extract fields from formalized contract
params = payload.get("params", {})
settings = payload.get("settings", {})
telemetry = payload.get("telemetry", {})

# Call handler with old-style params for backwards compatibility
# Handler will access flattened telemetry fields directly
result = execute(
    topic="",  # topic no longer in payload
    params=params,
    config=settings,  # settings replaces config
    telemetry=telemetry
)

# Handle error case
if isinstance(result, dict) and "error" in result:
    print(json.dumps({"error": result["error"]}))
else:
    # Convert flat dict result to new {text, html} format
    if isinstance(result, dict):
        # For now, format text from the dict and generate a basic HTML card
        text_parts = []
        for key, value in result.items():
            if key not in ("error",):
                if isinstance(value, (dict, list)):
                    continue
                text_parts.append(f"{key}: {value}")

        text = "\n".join(text_parts) if text_parts else json.dumps(result)

        # Generate inline HTML card from weather data
        # Use the handler's resolved location (from Open-Meteo or wttr.in),
        # not raw telemetry which may be stale or less accurate
        resolved_location = result.get("location", "")
        html = _generate_weather_html(result, resolved_location)

        print(json.dumps({"text": text, "html": html}))
    else:
        print(json.dumps({"text": str(result), "html": ""}))
