function fmtTime(ts) {
  try {
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch (_) {
    return "--:--:--";
  }
}

export function EventLog({ items, emptyText = "No events yet." }) {
  return (
    <div className="event-log">
      {items.length ? (
        items.map((item) => (
          <div key={item.id} className={`event-row tone-${item.tone || "neutral"}`}>
            <code>{fmtTime(item.ts)}</code>
            <strong>{item.title}</strong>
            {item.detail ? <span>{item.detail}</span> : null}
          </div>
        ))
      ) : (
        <div className="empty-card">{emptyText}</div>
      )}
    </div>
  );
}
