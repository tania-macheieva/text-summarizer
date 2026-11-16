import { NavLink, Route, Routes, Link } from "react-router-dom";
import SummarizePage from "./pages/SummarizePage";
import EvaluationPage from "./pages/EvaluationPage";
import AboutPage from "./pages/AboutPage";

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <Link className="brand" to="/" aria-label="Briefly — home">
          <span className="brand-mark" aria-hidden="true">s/</span>
          <span className="brand-copy">
            <strong>Briefly</strong>
            <span>text laboratory</span>
          </span>
        </Link>
        <nav>
          <NavLink to="/" end>Summarizer</NavLink>
          <NavLink to="/evaluation">Evaluation</NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
        <span className="topbar-meta">Text laboratory</span>
      </header>
      <main className="page-shell">
        <Routes>
          <Route path="/" element={<SummarizePage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </main>
    </div>
  );
}
