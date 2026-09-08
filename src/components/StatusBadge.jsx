import React from "react";

export default function StatusBadge({ status }) {
  const map = {
    checking: { text: "Checking backend...", cls: "status-checking" },
    online: { text: "Backend online", cls: "status-online" },
    offline: { text: "Backend offline \u2014 demo mode", cls: "status-offline" },
  };
  const s = map[status] || map.checking;
  return <div className={`status-badge ${s.cls}`}>{s.text}</div>;
}
