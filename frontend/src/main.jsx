import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { RealTest } from "./pages/RealTest";
import { Landing } from "./pages/Landing";
import "./styles.css";
import "leaflet/dist/leaflet.css";

function RootRouter() {
  const [path, setPath] = React.useState(window.location.pathname);

  React.useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  if (path === "/real-test") return <RealTest />;
  if (path === "/sim") return <App />;
  if (path === "/") return <Landing />;
  return <Landing />;
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RootRouter />
  </React.StrictMode>,
);
