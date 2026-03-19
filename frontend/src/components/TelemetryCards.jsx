function MetricCard({ label, value, tone = "default", note = "" }) {
  return (
    <div className={`metric metric-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {note ? <small>{note}</small> : null}
    </div>
  );
}

export function TelemetryCards({ items, emptyText = "Waiting for telemetry..." }) {
  const hasValue = items.some((item) => item.value !== "--");
  if (!hasValue) {
    return <div className="empty-card">{emptyText}</div>;
  }
  return (
    <div className="telemetry-grid">
      {items.map((item) => (
        <MetricCard
          key={item.label}
          label={item.label}
          value={item.value}
          tone={item.tone}
          note={item.note}
        />
      ))}
    </div>
  );
}
