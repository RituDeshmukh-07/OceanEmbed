import React from "react";
import Plot from "react-plotly.js";

export default function ProfileChart({ result }) {
  if (!result) {
    return (
      <div className="panel">
        <div className="panel-title">Predicted temperature profile</div>
        <div className="empty-state">Select a location and run reconstruction to see the profile.</div>
      </div>
    );
  }

  const { depths, predicted, referenceProfile } = result;
  const traces = [
    {
      x: predicted,
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "OceanEmbed prediction",
      line: { color: "#E8A33D", width: 2 },
      marker: { color: "#E8A33D", size: 5 },
    },
  ];
  if (referenceProfile) {
    traces.push({
      x: referenceProfile,
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "Reference profile",
      line: { color: "#4FB6C4", width: 2, dash: "dot" },
      marker: { color: "#4FB6C4", size: 5 },
    });
  }

  return (
    <div className="panel">
      <div className="panel-title">Predicted temperature profile</div>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 380,
          margin: { l: 55, r: 20, t: 10, b: 45 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#EDF2F4", size: 12 },
          xaxis: { title: "Temperature (\u00b0C)", gridcolor: "#22343F", zeroline: false },
          yaxis: { title: "Depth (m)", autorange: "reversed", gridcolor: "#22343F", zeroline: false },
          legend: { orientation: "h", y: -0.2 },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
