import React from "react";

const FIELDS = [
  { key: "SST", label: "SST", unit: "\u00b0C" },
  { key: "SSS", label: "SSS", unit: "PSU" },
  { key: "SSH", label: "SSH / SLA", unit: "m" },
  { key: "U_cur", label: "Current U", unit: "m/s" },
  { key: "V_cur", label: "Current V", unit: "m/s" },
  { key: "U_wind", label: "Wind U", unit: "m/s" },
  { key: "V_wind", label: "Wind V", unit: "m/s" },
];

export default function SurfaceVariables({ surface }) {
  return (
    <div className="panel">
      <div className="panel-title">Surface observations</div>
      <div className="surface-grid">
        {FIELDS.map((f) => {
          const val = surface ? surface[f.key] : null;
          return (
            <div className="surface-cell" key={f.key}>
              <div className="surface-label">{f.label}</div>
              <div className={val === null || val === undefined ? "surface-value na" : "surface-value"}>
                {val === null || val === undefined ? "Data unavailable" : `${val.toFixed(2)} ${f.unit}`}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
