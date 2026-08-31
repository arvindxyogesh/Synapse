import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Dashboard" },
  { to: "/playground", label: "Playground" },
  { to: "/requests", label: "Requests" },
  { to: "/api-keys", label: "API Keys" },
];

export default function Nav() {
  return (
    <nav className="flex items-center gap-6 border-b border-slate-800 px-6 py-4">
      <span className="font-semibold tracking-tight text-slate-50">Synapse</span>
      <div className="flex gap-4 text-sm">
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
