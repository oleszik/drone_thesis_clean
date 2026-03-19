export function ControlSection({ title, hint = "", children }) {
  return (
    <section className="control-section">
      <div className="control-section-head">
        <h3>{title}</h3>
        {hint ? <p className="hint">{hint}</p> : null}
      </div>
      <div className="control-section-body">{children}</div>
    </section>
  );
}
