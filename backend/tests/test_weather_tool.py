"""
Weather tool smoke — call execute() with a plain country name and assert the
result does not carry an error. Covers the contract the nightly scenario
asserts at the tool-routing level: the tool itself must successfully resolve
a location like 'germany' via the wttr.in fallback path (Open-Meteo requires
coordinates, which we don't supply here).

Marked @pytest.mark.integration because it makes a real outbound HTTP call
to wttr.in. Fast enough to run in the normal suite; isolated from the ACT
loop and the ONNX stack.
"""

import pytest

from tools import weather


@pytest.mark.integration
def test_weather_germany_returns_without_error():
    result = weather.execute(topic='test', params={'location': 'germany'})
    assert 'error' not in result, result
