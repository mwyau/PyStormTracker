(() => {
  "use strict";

  const DATA_URL = "tracks.trackjson";
  const HOUR_MS = 3_600_000;
  const SPEEDS = [0.25, 0.5, 1, 2, 4];
  const STEPS_PER_SECOND = 12;
  const TURBO_SCALE = [
    [0.0, [48, 18, 59]],
    [0.1, [70, 74, 164]],
    [0.2, [61, 126, 237]],
    [0.3, [33, 176, 213]],
    [0.4, [24, 215, 160]],
    [0.5, [100, 240, 91]],
    [0.6, [178, 230, 54]],
    [0.7, [238, 182, 48]],
    [0.8, [254, 112, 28]],
    [0.9, [226, 45, 6]],
    [1.0, [158, 1, 1]],
  ];

  const elements = {
    controls: document.getElementById("controls"),
    loading: document.getElementById("loading"),
    error: document.getElementById("error"),
    projection: document.getElementById("projection"),
    strength: document.getElementById("strength"),
    strengthLabel: document.getElementById("strength-label"),
    strengthValue: document.getElementById("strength-value"),
    duration: document.getElementById("duration"),
    durationValue: document.getElementById("duration-value"),
    displacement: document.getElementById("displacement"),
    displacementValue: document.getElementById("displacement-value"),
    timeStart: document.getElementById("time-start"),
    timeEnd: document.getElementById("time-end"),
    timeScrub: document.getElementById("time-scrub"),
    timeStartValue: document.getElementById("time-start-value"),
    timeCurrentValue: document.getElementById("time-current-value"),
    timeEndValue: document.getElementById("time-end-value"),
    playPause: document.getElementById("play-pause"),
    loop: document.getElementById("loop"),
    speedDown: document.getElementById("speed-down"),
    speed: document.getElementById("speed"),
    speedUp: document.getElementById("speed-up"),
    sameColor: document.getElementById("same-color"),
    reset: document.getElementById("reset"),
    legendHigh: document.getElementById("legend-high"),
    legendLow: document.getElementById("legend-low"),
    legendUnit: document.getElementById("legend-unit"),
    diagnostics: document.getElementById("diagnostics"),
  };

  const state = {
    data: null,
    map: null,
    overlay: null,
    filteredSegments: [],
    currentTime: 0,
    playing: false,
    speedIndex: 2,
    lastFrame: 0,
    frameCount: 0,
    fpsStart: 0,
    frontCenter: null,
    frontVersion: 0,
    frontUpdateQueued: false,
    filterExtension: null,
  };

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function requireArray(value, name) {
    if (!Array.isArray(value)) {
      throw new Error(`TrackJSON ${name} must be an array.`);
    }
    return value;
  }

  function requireFinite(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`TrackJSON ${name} must contain finite numbers.`);
    }
    return value;
  }

  function colorForValue(value, minimum, maximum, mode) {
    let position = maximum === minimum ? 0.5 : (value - minimum) / (maximum - minimum);
    position = Math.max(0, Math.min(1, position));
    if (mode === "min") position = 1 - position;
    for (let index = 0; index < TURBO_SCALE.length - 1; index += 1) {
      const lower = TURBO_SCALE[index];
      const upper = TURBO_SCALE[index + 1];
      if (position >= lower[0] && position <= upper[0]) {
        const fraction = (position - lower[0]) / (upper[0] - lower[0]);
        return [
          Math.round(lower[1][0] + (upper[1][0] - lower[1][0]) * fraction),
          Math.round(lower[1][1] + (upper[1][1] - lower[1][1]) * fraction),
          Math.round(lower[1][2] + (upper[1][2] - lower[1][2]) * fraction),
          220,
        ];
      }
    }
    return [...TURBO_SCALE.at(-1)[1], 220];
  }

  function greatCircleKm(lat1, lon1, lat2, lon2) {
    // Matches pystormtracker.models.geo.geod_dist_km and R_EARTH_KM.
    const radians = Math.PI / 180;
    const phi1 = lat1 * radians;
    const phi2 = lat2 * radians;
    const lambda1 = lon1 * radians;
    const lambda2 = lon2 * radians;
    const dot =
      Math.sin(phi1) * Math.sin(phi2) +
      Math.cos(phi1) * Math.cos(phi2) * Math.cos(lambda1 - lambda2);
    return 6371.22 * Math.acos(Math.max(-1, Math.min(1, dot)));
  }

  function prepareTrackJSON(document) {
    if (!isObject(document) || document.format !== "TrackJSON/1.0") {
      throw new Error("Unsupported input: the explorer requires TrackJSON/1.0 data.");
    }
    if (!isObject(document.metadata) || !isObject(document.points)) {
      throw new Error("TrackJSON requires metadata and points objects.");
    }
    const primaryVar = document.metadata.primary_var;
    const mode = document.metadata.mode;
    if (typeof primaryVar !== "string" || !primaryVar) {
      throw new Error("TrackJSON metadata.primary_var must identify the displayed variable.");
    }
    if (mode !== "min" && mode !== "max") {
      throw new Error("TrackJSON metadata.mode must be 'min' or 'max'.");
    }
    if (!isObject(document.points.variables)) {
      throw new Error("TrackJSON points.variables must be an object.");
    }
    const lats = requireArray(document.points.lat, "points.lat");
    const lons = requireArray(document.points.lon, "points.lon");
    const times = requireArray(document.points.time, "points.time");
    const values = requireArray(document.points.variables[primaryVar], `points.variables.${primaryVar}`);
    const rawTracks = requireArray(document.tracks, "tracks");
    if (lats.length !== lons.length || lats.length !== times.length || lats.length !== values.length) {
      throw new Error("TrackJSON coordinate and primary-variable arrays must have equal lengths.");
    }
    const finiteValues = values.filter((value) => typeof value === "number" && Number.isFinite(value));
    if (finiteValues.length === 0) {
      throw new Error("TrackJSON primary-variable values contain no finite values.");
    }
    const minimum = Math.min(...finiteValues);
    const maximum = Math.max(...finiteValues);
    const rawUnit = isObject(document.metadata.units) && typeof document.metadata.units[primaryVar] === "string"
      ? document.metadata.units[primaryVar]
      : "";
    const mslFixedPoint = primaryVar.toLowerCase() === "msl" && rawUnit === "Pa" && Math.max(...finiteValues.map(Math.abs)) >= 100_000;
    const displayScale = mslFixedPoint ? 1e-4 : 1;
    const unit = mslFixedPoint ? "hPa" : rawUnit;
    const tracks = [];
    const segments = [];
    for (const rawTrack of rawTracks) {
      if (!isObject(rawTrack) || !Number.isInteger(rawTrack.track_id)) {
        throw new Error("Each TrackJSON track must have an integer track_id.");
      }
      const { start, end } = rawTrack;
      if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end >= lats.length) {
        throw new Error("Each TrackJSON track must span at least one valid segment.");
      }
      const trackLats = lats.slice(start, end + 1).map((value) => requireFinite(value, "points.lat"));
      const trackLons = lons.slice(start, end + 1).map((value) => requireFinite(value, "points.lon"));
      const trackTimes = times.slice(start, end + 1).map((value) => requireFinite(value, "points.time"));
      const trackValues = values.slice(start, end + 1).map((value) => requireFinite(value, `points.variables.${primaryVar}`));
      const peakValue = mode === "min" ? Math.min(...trackValues) : Math.max(...trackValues);
      const track = {
        id: rawTrack.track_id,
        start,
        end,
        peakValue,
        displayPeakValue: peakValue * displayScale,
        durationHours: (trackTimes.at(-1) - trackTimes[0]) / HOUR_MS,
        displacementKm: greatCircleKm(trackLats[0], trackLons[0], trackLats.at(-1), trackLons.at(-1)),
      };
      tracks.push(track);
      for (let target = start + 1; target <= end; target += 1) {
        const endpointValue = requireFinite(values[target], `points.variables.${primaryVar}`);
        const endpointTime = requireFinite(times[target], "points.time");
        segments.push({
          track,
          sourcePosition: [requireFinite(lons[target - 1], "points.lon"), requireFinite(lats[target - 1], "points.lat")],
          targetPosition: [requireFinite(lons[target], "points.lon"), requireFinite(lats[target], "points.lat")],
          endpointValue,
          endpointTime,
          color: colorForValue(endpointValue, minimum, maximum, mode),
        });
      }
    }
    const finiteTimes = times.filter((value) => typeof value === "number" && Number.isFinite(value));
    const minTime = Math.min(...finiteTimes);
    for (const segment of segments) segment.filterTime = segment.endpointTime - minTime;
    const strengthMinimum = Math.min(...tracks.map((track) => track.displayPeakValue));
    const strengthMaximum = Math.max(...tracks.map((track) => track.displayPeakValue));
    return {
      primaryVar,
      mode,
      unit,
      minimum,
      maximum,
      displayScale,
      strengthMinimum,
      strengthMaximum,
      tracks,
      segments,
      minTime,
      maxTime: Math.max(...finiteTimes),
    };
  }

  function formatTime(time) {
    return new Date(time).toISOString().slice(0, 16).replace("T", " ");
  }

  function formatValue(value) {
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
  }

  function rgba(color) {
    return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${color[3] / 255})`;
  }

  function escaped(value) {
    return String(value).replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]);
  }

  function timeStep(data) {
    const sorted = [...new Set(data.segments.map((segment) => segment.endpointTime))].sort((a, b) => a - b);
    let step = HOUR_MS * 6;
    for (let index = 1; index < sorted.length; index += 1) {
      const candidate = sorted[index] - sorted[index - 1];
      if (candidate > 0) step = Math.min(step, candidate);
    }
    return step;
  }

  function currentFilter() {
    return {
      strength: Number(elements.strength.value),
      duration: Number(elements.duration.value),
      displacement: Number(elements.displacement.value),
    };
  }

  function trackMatches(track, filter) {
    const strengthMatches = state.data.mode === "min"
      ? track.displayPeakValue <= filter.strength
      : track.displayPeakValue >= filter.strength;
    return strengthMatches && track.durationHours >= filter.duration && track.displacementKm >= filter.displacement;
  }

  function updateLabels() {
    const filter = currentFilter();
    elements.strengthValue.value = `${formatValue(filter.strength)} ${state.data.unit}`.trim();
    elements.durationValue.value = `${formatValue(filter.duration)} h`;
    elements.displacementValue.value = `${formatValue(filter.displacement)} km`;
    elements.timeStartValue.value = formatTime(Number(elements.timeStart.value));
    elements.timeCurrentValue.value = formatTime(state.currentTime);
    elements.timeEndValue.value = formatTime(Number(elements.timeEnd.value));
    elements.speed.value = `${SPEEDS[state.speedIndex]}×`;
  }

  function setLayers(staticDataChanged = false) {
    const start = Number(elements.timeStart.value);
    const end = state.currentTime;
    const filterRange = [start - state.data.minTime, end - state.data.minTime];
    const properties = {
      id: "track-segments",
      data: state.filteredSegments,
      pickable: true,
      getSourcePosition: (segment) => segment.sourcePosition,
      getTargetPosition: (segment) => segment.targetPosition,
      getColor: (segment) => {
        if (elements.projection.value === "globe" && !isFrontHemisphere(segment, state.frontCenter)) {
          return [segment.color[0], segment.color[1], segment.color[2], 0];
        }
        return segment.color;
      },
      getWidth: 2,
      widthUnits: "pixels",
      widthMinPixels: 1.5,
      parameters: { cullMode: "none", depthTest: false },
      extensions: [state.filterExtension],
      getFilterValue: (segment) => segment.filterTime,
      filterRange,
      updateTriggers: { getColor: state.frontVersion },
    };
    const pointProperties = {
      id: "track-points",
      data: state.filteredSegments,
      pickable: true,
      getPosition: (segment) => segment.targetPosition,
      getFillColor: (segment) => [segment.color[0], segment.color[1], segment.color[2], 0],
      getRadius: 0.1,
      radiusUnits: "pixels",
      radiusMaxPixels: 0.1,
      extensions: [state.filterExtension],
      getFilterValue: (segment) => segment.filterTime,
      filterRange,
    };
    state.overlay.setProps({
      layers: [new deck.LineLayer(properties), new deck.ScatterplotLayer(pointProperties)],
      getTooltip: ({ object }) => {
        if (!object) return null;
        const track = object.track;
        return {
          html: `<strong>Track ${escaped(track.id)}</strong><br>${escaped(state.data.primaryVar)}: ${escaped(formatValue(object.endpointValue * state.data.displayScale))} ${escaped(state.data.unit)}<br>Peak: ${escaped(formatValue(track.displayPeakValue))} ${escaped(state.data.unit)}<br>Duration: ${escaped(formatValue(track.durationHours))} h<br>Displacement: ${escaped(formatValue(track.displacementKm))} km<br>${escaped(formatTime(object.endpointTime))}`,
          style: { backgroundColor: rgba(object.color), color: "#fff" },
        };
      },
    });
    updateLabels();
    void staticDataChanged;
  }

  function rebuildStaticData() {
    const filter = currentFilter();
    const sameColor = elements.sameColor.checked;
    state.filteredSegments = state.data.segments
      .filter((segment) => trackMatches(segment.track, filter))
      .map((segment) => {
        if (!sameColor) return segment;
        return {
          ...segment,
          color: colorForValue(segment.track.peakValue, state.data.minimum, state.data.maximum, state.data.mode),
        };
      });
    updateFrontHemisphere(true);
  }

  function isFrontHemisphere(segment, center) {
    if (center === null) return true;
    const longitude1 = segment.sourcePosition[0] * Math.PI / 180;
    const latitude1 = segment.sourcePosition[1] * Math.PI / 180;
    const longitude2 = segment.targetPosition[0] * Math.PI / 180;
    const latitude2 = segment.targetPosition[1] * Math.PI / 180;
    const midpointLongitude = longitude1 + Math.atan2(Math.sin(longitude2 - longitude1), Math.cos(longitude2 - longitude1)) / 2;
    const midpointLatitude = (latitude1 + latitude2) / 2;
    const centerLongitude = center.lng * Math.PI / 180;
    const centerLatitude = center.lat * Math.PI / 180;
    const dot =
      Math.sin(midpointLatitude) * Math.sin(centerLatitude) +
      Math.cos(midpointLatitude) * Math.cos(centerLatitude) * Math.cos(midpointLongitude - centerLongitude);
    return dot >= 0;
  }

  function updateFrontHemisphere(staticDataChanged = false) {
    state.frontCenter = state.map.getCenter();
    state.frontVersion += 1;
    setLayers(staticDataChanged);
  }

  function scheduleFrontHemisphereUpdate() {
    if (state.frontUpdateQueued) return;
    state.frontUpdateQueued = true;
    requestAnimationFrame(() => {
      state.frontUpdateQueued = false;
      updateFrontHemisphere(false);
    });
  }

  function stopPlayback() {
    state.playing = false;
    elements.playPause.textContent = "Play";
  }

  function animationFrame(timestamp) {
    if (!state.playing) return;
    if (state.lastFrame === 0) state.lastFrame = timestamp;
    const interval = 1000 / (STEPS_PER_SECOND * SPEEDS[state.speedIndex]);
    if (timestamp - state.lastFrame >= interval) {
      const next = state.currentTime + Number(elements.timeScrub.step);
      const last = Number(elements.timeEnd.value);
      if (next > last) {
        if (elements.loop.checked) {
          state.currentTime = Number(elements.timeStart.value);
        } else {
          state.currentTime = last;
          stopPlayback();
        }
      } else {
        state.currentTime = next;
      }
      elements.timeScrub.value = String(state.currentTime);
      setLayers(false);
      state.lastFrame = timestamp;
    }
    requestAnimationFrame(animationFrame);
  }

  function measureFrames(timestamp) {
    state.frameCount += 1;
    if (timestamp - state.fpsStart >= 1000) {
      elements.diagnostics.value = `FPS: ${state.frameCount}`;
      state.frameCount = 0;
      state.fpsStart = timestamp;
    }
    requestAnimationFrame(measureFrames);
  }

  function resetControls() {
    const { data } = state;
    elements.projection.value = "globe";
    const defaultStrength = data.primaryVar.toLowerCase() === "msl" && data.unit === "hPa"
      ? Math.max(data.strengthMinimum, Math.min(data.strengthMaximum, -15))
      : data.mode === "min" ? data.strengthMaximum : data.strengthMinimum;
    elements.strength.value = String(defaultStrength);
    elements.duration.value = "48";
    elements.displacement.value = String(Math.min(1000, Number(elements.displacement.max)));
    elements.timeStart.value = String(data.minTime);
    elements.timeEnd.value = String(data.maxTime);
    elements.timeScrub.min = String(data.minTime);
    elements.timeScrub.max = String(data.maxTime);
    state.currentTime = data.maxTime;
    elements.timeScrub.value = String(state.currentTime);
    elements.sameColor.checked = false;
    elements.loop.checked = false;
    state.speedIndex = 2;
    stopPlayback();
    state.map.setProjection({ type: "globe" });
    rebuildStaticData();
  }

  function bindControls() {
    const updateStatic = () => {
      stopPlayback();
      rebuildStaticData();
    };
    for (const element of [elements.strength, elements.duration, elements.displacement, elements.sameColor]) {
      element.addEventListener("input", updateStatic);
      element.addEventListener("change", updateStatic);
    }
    const updateTime = () => {
      stopPlayback();
      if (Number(elements.timeStart.value) > Number(elements.timeEnd.value)) elements.timeEnd.value = elements.timeStart.value;
      elements.timeScrub.min = elements.timeStart.value;
      elements.timeScrub.max = elements.timeEnd.value;
      state.currentTime = Math.min(Math.max(state.currentTime, Number(elements.timeStart.value)), Number(elements.timeEnd.value));
      elements.timeScrub.value = String(state.currentTime);
      setLayers(false);
    };
    elements.timeStart.addEventListener("input", updateTime);
    elements.timeEnd.addEventListener("input", updateTime);
    elements.timeScrub.addEventListener("input", () => {
      stopPlayback();
      state.currentTime = Number(elements.timeScrub.value);
      setLayers(false);
    });
    elements.projection.addEventListener("change", () => {
      state.map.setProjection({ type: elements.projection.value });
      updateFrontHemisphere(false);
    });
    elements.playPause.addEventListener("click", () => {
      state.playing = !state.playing;
      elements.playPause.textContent = state.playing ? "Pause" : "Play";
      if (state.playing) {
        if (state.currentTime >= Number(elements.timeEnd.value)) {
          state.currentTime = Number(elements.timeStart.value);
          elements.timeScrub.value = String(state.currentTime);
          setLayers(false);
        }
        state.lastFrame = 0;
        requestAnimationFrame(animationFrame);
      }
    });
    elements.speedDown.addEventListener("click", () => {
      state.speedIndex = Math.max(0, state.speedIndex - 1);
      updateLabels();
    });
    elements.speedUp.addEventListener("click", () => {
      state.speedIndex = Math.min(SPEEDS.length - 1, state.speedIndex + 1);
      updateLabels();
    });
    elements.reset.addEventListener("click", resetControls);
  }

  function initializeControls(data) {
    const step = timeStep(data);
    const maximumDuration = Math.ceil(Math.max(...data.tracks.map((track) => track.durationHours)) / 6) * 6;
    const maximumDisplacement = Math.ceil(Math.max(...data.tracks.map((track) => track.displacementKm)) / 100) * 100;
    elements.strength.min = String(data.strengthMinimum);
    elements.strength.max = String(data.strengthMaximum);
    elements.strength.step = String((data.strengthMaximum - data.strengthMinimum) / 500 || 1);
    elements.strengthLabel.textContent = `Peak ${data.primaryVar} (${data.unit || "unit unavailable"})`;
    elements.duration.max = String(maximumDuration);
    elements.displacement.max = String(maximumDisplacement);
    for (const element of [elements.timeStart, elements.timeEnd, elements.timeScrub]) {
      element.min = String(data.minTime);
      element.max = String(data.maxTime);
      element.step = String(step);
    }
    const high = data.mode === "min" ? data.minimum : data.maximum;
    const low = data.mode === "min" ? data.maximum : data.minimum;
    elements.legendHigh.textContent = formatValue(high * data.displayScale);
    elements.legendLow.textContent = formatValue(low * data.displayScale);
    elements.legendUnit.textContent = `${data.primaryVar} ${data.unit}`.trim();
  }

  async function loadTrackJSON() {
    const response = await fetch(DATA_URL);
    if (!response.ok) throw new Error(`Could not fetch ${DATA_URL} (${response.status}).`);
    return response.json();
  }

  async function start() {
    try {
      if (!window.maplibregl || !window.deck) throw new Error("MapLibre GL or Deck.gl did not load from jsDelivr.");
      state.data = prepareTrackJSON(await loadTrackJSON());
      initializeControls(state.data);
      state.filterExtension = new deck.DataFilterExtension({ filterSize: 1 });
      state.map = new maplibregl.Map({
        container: "map",
        style: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        center: [-60, 35],
        zoom: 2,
        projection: { type: "globe" },
      });
      state.map.on("load", () => {
        state.overlay = new deck.MapboxOverlay({ interleaved: false, layers: [] });
        state.map.addControl(state.overlay);
        state.map.on("move", scheduleFrontHemisphereUpdate);
        state.map.on("moveend", () => updateFrontHemisphere(false));
        bindControls();
        state.fpsStart = performance.now();
        requestAnimationFrame(measureFrames);
        resetControls();
        elements.loading.hidden = true;
        elements.controls.hidden = false;
      });
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Unknown error";
      elements.loading.hidden = true;
      elements.error.textContent = `${detail} Supply an adjacent TrackJSON/1.0 file named ${DATA_URL}.`;
      elements.error.hidden = false;
    }
  }

  start();
})();
