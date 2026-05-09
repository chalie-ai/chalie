"""
Tool Output Utilities — Telemetry construction.

Shared helpers used by ToolRegistryService for telemetry construction.
"""

def build_tool_telemetry(raw_telemetry: dict) -> dict:
    """Flatten client context telemetry into the tool contract format.

    Extracts location, time, locale, language, and device context from
    the raw ClientContextService output.

    Args:
        raw_telemetry: Raw dict from ClientContextService.get()

    Returns:
        Flattened telemetry dict suitable for tool container env vars
    """
    loc = raw_telemetry.get("location") or {}
    loc_name = raw_telemetry.get("location_name", "")
    city, country = "", ""
    if "," in loc_name:
        city, country = [p.strip() for p in loc_name.split(",", 1)]

    device = raw_telemetry.get("device") or {}

    result = {
        "lat": loc.get("lat"),
        "lon": loc.get("lon"),
        "location_name": raw_telemetry.get("location_name", ""),
        "city": city,
        "country": country,
        "time": raw_telemetry.get("local_time", ""),
        "locale": raw_telemetry.get("locale", ""),
        "language": raw_telemetry.get("language", ""),
    }

    # Device context — so tools can tailor output to user's device
    if device_class := device.get("class"):
        result["device_class"] = device_class
    if platform := device.get("platform"):
        result["platform"] = platform
    if "pwa" in device:
        result["pwa"] = device["pwa"]

    return result
