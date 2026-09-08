import React, { useEffect, useState, useCallback } from "react";
import MapPanel from "./components/MapPanel";
import SurfaceVariables from "./components/SurfaceVariables";
import ProfileChart from "./components/ProfileChart";
import StatsPanel from "./components/StatsPanel";
import StatusBadge from "./components/StatusBadge";
import {
  checkHealth,
  fetchRegion,
  runPrediction,
  surfaceAtGrid,
  latLonToGrid,
  generateDemoProfile,
  DEMO_MODE_ENABLED,
} from "./api/oceanApi";

export default function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [regionData, setRegionData] = useState(null);
  const [selected, setSelected] = useState(null);
  const [date, setDate] = useState(new Date().toISOString().slice(0, 10));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [surface, setSurface] = useState(null);

  useEffect(() => {
    checkHealth()
      .then(() => setBackendStatus("online"))
      .catch(() => setBackendStatus("offline"));
    fetchRegion()
      .then(setRegionData)
      .catch(() => setRegionData(null));
  }, []);

  const handleSelect = useCallback(
    (lat, lon) => {
      setSelected({ lat, lon });
      setResult(null);
      setError("");
      if (backendStatus === "online" && regionData) {
        const { x, y } = latLonToGrid(lat, lon, regionData.grid_size);
        setSurface(surfaceAtGrid(regionData, x, y));
      } else {
        setSurface(null);
      }
    },
    [backendStatus, regionData]
  );

  const handleReconstruct = async () => {
    if (!selected) {
      setError("Select a location on the map first.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const gridSize = regionData?.grid_size ?? 32;
      const data = await runPrediction(selected.lat, selected.lon, date, gridSize);
      setResult(data);
      if (data.surface) {
        setSurface(data.surface);
      } else {
        const { x, y } = latLonToGrid(selected.lat, selected.lon, gridSize);
        setSurface(surfaceAtGrid(regionData, x, y));
      }
    } catch (e) {
      setError(`Prediction request failed: ${e.message}`);
      if (DEMO_MODE_ENABLED) {
        const demo = generateDemoProfile(selected.lat, selected.lon);
        setResult(demo);
        setSurface(demo.surface);
      } else {
        setResult(null);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>OceanEmbed</h1>
          <div className="subtitle">Subsurface Temperature Reconstruction &mdash; North Indian Ocean</div>
        </div>
        <StatusBadge status={backendStatus} />
      </header>

      {result?.isDemo && (
        <div className="demo-banner">DEMO MODE &mdash; SIMULATED DATA</div>
      )}

      <div className="layout">
        <div className="col">
          <MapPanel selected={selected} onSelect={handleSelect} />

          <div className="panel controls-panel">
            <div className="panel-title">Reconstruction controls</div>
            <label className="field-label" htmlFor="date-input">Observation date</label>
            <input
              id="date-input"
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="date-input"
            />
            <button className="reconstruct-btn" onClick={handleReconstruct} disabled={loading}>
              {loading ? "Reconstructing..." : "RECONSTRUCT TEMPERATURE"}
            </button>
            {error && <div className="error-msg">{error}</div>}
          </div>

          <SurfaceVariables surface={surface} />
        </div>

        <div className="col">
          <ProfileChart result={result} />
          <StatsPanel result={result} />
        </div>
      </div>
    </div>
  );
}
