# Interactive output

The Track Explorer is temporarily disabled while its implementation is being
redesigned. The current HTML output is a static placeholder and does not embed
TrackJSON data.

TrackJSON remains the native interchange format. Its canonical numeric time
columns, calendar metadata, optional declared bounds, and optional derived
statistics are specified in [TrackJSON v1.0](trackjson.md).

HTML conversion is retained for command-line compatibility and emits the same
placeholder. It does not create temporary TrackJSON files or inject data into
an executable page. A future explorer will use packed numeric times and
calendar-aware labels generated at the Python presentation boundary.
