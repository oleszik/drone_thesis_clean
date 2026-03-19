function StatusPill({ label, value, tone = "neutral", subtle = false }) {
  const cls = ["status-pill", `tone-${tone}`, subtle ? "subtle" : ""].filter(Boolean).join(" ");
  return (
    <div className={cls}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function StatusBar({ items, detail = "" }) {
  return (
    <section className="status-bar">
      <div className="status-pill-grid">
        {items.map((item) => (
          <StatusPill
            key={item.label}
            label={item.label}
            value={item.value}
            tone={item.tone}
            subtle={item.subtle}
          />
        ))}
      </div>
      {detail ? <p className="status-bar-detail">{detail}</p> : null}
    </section>
  );
}
