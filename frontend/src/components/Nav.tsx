import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Dashboard" },
  { to: "/playground", label: "Playground" },
  { to: "/requests", label: "Requests" },
  { to: "/api-keys", label: "API Keys" },
  { to: "/benchmarks", label: "Benchmarks" },
];

export default function Nav() {
  return (
    <nav className="flex items-center gap-6 border-b border-slate-800 px-4 py-4 sm:px-6">
      <span className="shrink-0 font-semibold tracking-tight text-slate-50">Synapse</span>
      {/* Scrolls sideways on narrow screens instead of wrapping or overflowing. */}
      <div className="flex gap-4 overflow-x-auto whitespace-nowrap text-sm">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.to === "/"}
            className={({ isActive }) =>
              isActive ? "text-emerald-400" : "text-slate-400 hover:text-slate-200"
            }
          >
            {link.label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
