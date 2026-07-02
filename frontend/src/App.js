import React, { useState, useEffect } from "react";
import Dashboard from "./pages/Dashboard";

function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("stress-theme") || "cyber";
  });

  useEffect(() => {
    localStorage.setItem("stress-theme", theme);
    document.body.className = `theme-${theme}`;
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => {
      if (prev === "cyber") return "earthy";
      if (prev === "earthy") return "clinical";
      return "cyber";
    });
  };

  return (
    <div className={`App theme-${theme}`}>
      <Dashboard theme={theme} toggleTheme={toggleTheme} />
    </div>
  );
}

export default App;
