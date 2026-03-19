import { useEffect, useState } from "react";

const BACKEND_BASE = import.meta.env.VITE_BACKEND_BASE || "http://127.0.0.1:8000";

export function useLiveStream(path, { event = null, enabled = true, resetKey = "", onMessage = null } = {}) {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!enabled || !path) {
      setConnected(false);
      return undefined;
    }

    const source = new EventSource(`${BACKEND_BASE}${path}`);

    const handleMessage = (msgEvent) => {
      try {
        const payload = JSON.parse(msgEvent.data);
        setConnected(true);
        setError(null);
        onMessage?.(payload);
      } catch (err) {
        setError(err);
      }
    };

    const listenerEvent = event || "message";
    if (event) source.addEventListener(event, handleMessage);
    else source.onmessage = handleMessage;

    source.onopen = () => {
      setConnected(true);
      setError(null);
    };

    source.onerror = () => {
      setConnected(false);
      setError(new Error(`stream error: ${path}`));
    };

    return () => {
      if (event) source.removeEventListener(listenerEvent, handleMessage);
      source.close();
      setConnected(false);
    };
  }, [enabled, event, onMessage, path, resetKey]);

  return { connected, error };
}
