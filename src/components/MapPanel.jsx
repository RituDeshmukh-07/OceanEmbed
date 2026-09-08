import React from "react";
import { MapContainer, TileLayer, Marker, Rectangle, useMapEvents } from "react-leaflet";
import L from "leaflet";
import { REGION_BOUNDS, isInRegion } from "../api/oceanApi";

// default Leaflet marker icons (bundled asset paths break under Vite otherwise)
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";
const markerIconInstance = L.icon({
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

const bounds = [
  [REGION_BOUNDS.latMin, REGION_BOUNDS.lonMin],
  [REGION_BOUNDS.latMax, REGION_BOUNDS.lonMax],
];

function ClickHandler({ onSelect, onInvalid }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      if (isInRegion(lat, lng)) {
        onSelect(lat, lng);
      } else {
        onInvalid();
      }
    },
  });
  return null;
}

export default function MapPanel({ selected, onSelect }) {
  const [invalidMsg, setInvalidMsg] = React.useState("");

  const handleInvalid = () => {
    setInvalidMsg("That point is outside the North Indian Ocean study region.");
    setTimeout(() => setInvalidMsg(""), 3000);
  };

  return (
    <div className="panel map-panel">
      <div className="panel-title">Region &mdash; North Indian Ocean</div>
      <MapContainer
        center={[17.5, 75]}
        zoom={4}
        minZoom={3}
        maxZoom={8}
        maxBounds={bounds}
        maxBoundsViscosity={1.0}
        style={{ height: "360px", width: "100%", borderRadius: "8px" }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
        />
        <Rectangle bounds={bounds} pathOptions={{ color: "#4FB6C4", weight: 1.5, fillOpacity: 0.03 }} />
        <ClickHandler onSelect={onSelect} onInvalid={handleInvalid} />
        {selected && (
          <Marker position={[selected.lat, selected.lon]} icon={markerIconInstance} />
        )}
      </MapContainer>
      {invalidMsg && <div className="map-warning">{invalidMsg}</div>}
      <div className="map-coords">
        {selected
          ? `Selected: ${selected.lat.toFixed(2)}\u00b0N, ${selected.lon.toFixed(2)}\u00b0E`
          : "Click within the highlighted box to select a location"}
      </div>
    </div>
  );
}
