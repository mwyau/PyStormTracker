# WebGL Explorer

The WebGL explorer displays TrackJSON v1 trajectories with Deck.gl and
MapLibre GL. It is an online viewer: its JavaScript libraries load from
jsDelivr and its CARTO Positron basemap loads at runtime.

<iframe src="_static/explorer/explorer.html" width="100%" height="1000" title="PyStormTracker WebGL explorer" style="border: 1px solid #d8dfe6; border-radius: 8px;"></iframe>

If the embedded viewer is unavailable, <a href="_static/explorer/explorer.html" target="_blank">open the explorer in a new tab</a>.

The documentation build copies the ERA5 DJF 2025–2026 Hodges TrackJSON test
fixture directly into its output directory. It contains 20,099 trajectory
segments.

## Input requirements

Place `explorer.html`, `explorer.css`, `explorer.js`, and a TrackJSON v1 file
named `tracks.trackjson` in the same directory. The viewer checks the format,
metadata, point arrays, primary variable, and track ranges before rendering.
It reports a corrective error when the file is not supported.

Colors use each segment endpoint's primary-variable value. For a
minimum-tracked variable such as mean sea level pressure, lower values map to
the high end of the color scale; maximum-tracked variables use the normal
order. The **Same color per track** option instead uses the primary-variable
peak derived from each track's points. Popups use the selected segment's
rendered color.

The controls filter by peak strength, duration, displacement, and time. The
initial view shows all selected trajectories. Playback restarts at the selected
start time and accumulates segments until the selected end time. During
playback, Deck.gl updates only the GPU time-filter uniform and the upper-right
diagnostic shows the current frame rate. Hardware frame rate is intentionally
not asserted in automated tests.
