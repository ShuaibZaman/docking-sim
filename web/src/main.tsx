import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import ComparisonApp from "./ComparisonApp";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ComparisonApp />
  </StrictMode>,
);
