import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import StationMap from './components/StationMap';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>Range Guardian</h1>
          <nav>
            <ul>
              <li>
                <Link to="/">Dashboard</Link>
              </li>
              <li>
                <Link to="/map">Find Stations</Link>
              </li>
            </ul>
          </nav>
        </header>
        
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/map" element={<StationMap />} />
          </Routes>
        </main>
        
        <footer>
          <p>&copy; 2023 Range Guardian</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;