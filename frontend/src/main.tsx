import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, HashRouter } from "react-router-dom";

import App from "./App";
import { STATIC_DEMO } from "./demo";

// GitHub Pages serves files only, so a deep link like /playground would 404
// there; the static demo uses #/playground-style URLs instead.
const Router = STATIC_DEMO ? HashRouter : BrowserRouter;
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Router>
      <App />
    </Router>
  </React.StrictMode>,
);
