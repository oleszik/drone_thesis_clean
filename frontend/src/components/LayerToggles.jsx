export function LayerToggles({ items, legend }) {
  return (
    <div className="map-ops-grid">
      <section className="map-sidecard">
        <div className="map-sidecard-head">
          <h3>Layers</h3>
          <p className="hint">Show only what helps the operator right now.</p>
        </div>
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
      </section>

      <section className="map-sidecard">
        <div className="map-sidecard-head">
          <h3>Legend</h3>
          <p className="hint">Map colors and symbols used during operation.</p>
        </div>
        <div className="legend-list">
          {legend.map((item) => (
            <div key={item.label} className="legend-row">
              <span className={`legend-swatch ${item.swatchClass || ""}`}>{item.symbol || ""}</span>
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
