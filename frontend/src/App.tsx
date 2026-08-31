import { Route, Routes } from "react-router-dom";

import Nav from "./components/Nav";
import ApiKeys from "./pages/ApiKeys";
import Dashboard from "./pages/Dashboard";
import Requests from "./pages/Requests";

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950">
      <Nav />
      <main className="mx-auto max-w-6xl px-6 py-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/requests" element={<Requests />} />
          <Route path="/api-keys" element={<ApiKeys />} />
        </Routes>
      </main>
    </div>
  );
}
