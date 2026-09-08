import React from "react";

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value === null || value === undefined || Number.isNaN(value) ? "N/A" : value.toFixed(3)}</div>
    </div>
  );
}

export default function StatsPanel({ result }) {
  return (
    <div className="panel">
      <div className="panel-title">Skill metrics</div>
      <div className="stats-row">
        <StatCard label="RMSE (\u00b0C)" value={result?.rmse ?? null} />
        <StatCard label="Bias (\u00b0C)" value={result?.bias ?? null} />
        <StatCard label="Correlation" value={result?.correlation ?? null} />
      </div>
    </div>
  );
}
