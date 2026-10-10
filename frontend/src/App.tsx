import { Route, Routes } from "react-router-dom";

import Nav from "./components/Nav";
import { REPO_URL, STATIC_DEMO } from "./demo";
import ApiKeys from "./pages/ApiKeys";
import Benchmarks from "./pages/Benchmarks";
import Dashboard from "./pages/Dashboard";
import Playground from "./pages/Playground";
import ReplayPlayground from "./pages/ReplayPlayground";
import Requests from "./pages/Requests";

function NeedsLiveGateway() {
  return (
    <div className="max-w-xl rounded-lg border border-slate-800 bg-slate-900 p-4 text-sm text-slate-300">
      This page shows live data from a running gateway, which this static demo doesn't have.{" "}
      <a className="text-emerald-400 hover:underline" href={REPO_URL}>Run Synapse yourself</a> to see it, or look at the
      Benchmarks and the replay Playground.
    </div>
  );
}

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950">
      <Nav />
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        {STATIC_DEMO ? (
          <Routes>
            <Route path="/" element={<Benchmarks />} />
            <Route path="/benchmarks" element={<Benchmarks />} />
            <Route path="/playground" element={<ReplayPlayground />} />
            <Route path="*" element={<NeedsLiveGateway />} />
          </Routes>
        ) : (
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/requests" element={<Requests />} />
            <Route path="/api-keys" element={<ApiKeys />} />
            <Route path="/benchmarks" element={<Benchmarks />} />
          </Routes>
        )}
      </main>
    </div>
  );
}
