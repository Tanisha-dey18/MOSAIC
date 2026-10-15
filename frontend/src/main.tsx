import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import Layout from "./components/Layout";
import About from "./pages/About";
import Baselines from "./pages/Baselines";
import Dashboard from "./pages/Dashboard";
import DataExplorer from "./pages/DataExplorer";
import Explainability from "./pages/Explainability";
import ModelLab from "./pages/ModelLab";
import Monitoring from "./pages/Monitoring";
import NotFound from "./pages/NotFound";
import Reports from "./pages/Reports";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="data" element={<DataExplorer />} />
          <Route path="monitoring" element={<Monitoring />} />
          <Route path="baselines" element={<Baselines />} />
          <Route path="models" element={<ModelLab />} />
          <Route path="explain" element={<Explainability />} />
          <Route path="reports" element={<Reports />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
