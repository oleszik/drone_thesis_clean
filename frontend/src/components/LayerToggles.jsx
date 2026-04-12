export function LayerToggles({ items, legend }) {
  return (
    <div className="map-ops-grid">
      <details className="map-sidecard map-sidecard-collapsible" open>
        <summary className="map-sidecard-head">
          <h3>Layers</h3>
          <p className="hint">Toggle overlays</p>
        </summary>
        <div className="toggle-grid">
          {items.map((item) => (
            <label key={item.key} className="toggle-row">
              <input
                type="checkbox"
                checked={Boolean(item.checked)}
                onChange={(e) => item.onChange(e.target.checked)}
              />
              <span>{item.label}</span>
            </label>
          ))}
        </div>
      </details>

      <details className="map-sidecard map-sidecard-collapsible">
        <summary className="map-sidecard-head">
          <h3>Legend</h3>
          <p className="hint">Map symbols</p>
        </summary>
        <div className="legend-list">
          {legend.map((item) => (
            <div key={item.label} className="legend-row">
              <span className={`legend-swatch ${item.swatchClass || ""}`}>{item.symbol || ""}</span>
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}
