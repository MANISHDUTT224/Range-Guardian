import React, { useState, useEffect } from 'react';

const Dashboard = () => {
  const [batteryLevel, setBatteryLevel] = useState(85);
  const [alertStatus, setAlertStatus] = useState('normal');
  
  // Simulate battery drain
  useEffect(() => {
    const interval = setInterval(() => {
      setBatteryLevel(prevLevel => {
        const newLevel = prevLevel - 0.1;
        
        // Update alert status based on battery level
        if (newLevel <= 20) {
          setAlertStatus('critical');
        } else if (newLevel <= 30) {
          setAlertStatus('warning');
        } else {
          setAlertStatus('normal');
        }
        
        return newLevel > 0 ? newLevel : 0;
      });
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="dashboard">
      <h2>Battery Status</h2>
      <div className={`battery-indicator ${alertStatus}`}>
        <div 
          className="battery-level" 
          style={{ width: `${batteryLevel}%` }}
        ></div>
        <span className="battery-text">{Math.round(batteryLevel)}%</span>
      </div>
      
      {alertStatus === 'warning' && (
        <div className="alert warning">
          Warning: Battery below 30%. Consider finding a charging station soon.
        </div>
      )}
      
      {alertStatus === 'critical' && (
        <div className="alert critical">
          CRITICAL: Battery below 20%. Find a charging station immediately!
        </div>
      )}
    </div>
  );
};

export default Dashboard;