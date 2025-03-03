import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';
// import axios from 'axios';

// Fix for Leaflet marker icons
import L from 'leaflet';

delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

const StationMap = () => {
  const [stations, setStations] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    // For initial development, use mock data
    // Later, replace with actual API call
    const mockStations = [
      {
        id: "station1",
        name: "City Center Station",
        type: "public",
        status: "available",
        cost: 2.50,
        latitude: 37.7749,
        longitude: -122.4194
      },
      {
        id: "station2",
        name: "Park Station",
        type: "public",
        status: "busy",
        cost: 1.75,
        latitude: 37.7850,
        longitude: -122.4064
      },
      {
        id: "home1",
        name: "Mike's Home Charger",
        type: "home",
        status: "available",
        cost: 3.00,
        latitude: 37.7833,
        longitude: -122.4167
      }
    ];
    
    setStations(mockStations);
    setIsLoading(false);
    
    // Uncomment when backend is ready:
    /*
    axios.get('http://localhost:8000/api/v1/stations/charging-stations')
      .then(response => {
        setStations(response.data);
        setIsLoading(false);
      })
      .catch(error => {
        console.error('Error fetching stations:', error);
        setIsLoading(false);
      });
    */
  }, []);
  
  // Default map center (San Francisco)
  const defaultCenter = [37.7749, -122.4194];
  
  if (isLoading) {
    return <div>Loading map...</div>;
  }
  
  return (
    <div className="station-map">
      <h2>Charging Stations</h2>
      <MapContainer 
        center={defaultCenter} 
        zoom={13} 
        style={{ height: '400px', width: '100%' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        {stations.map(station => (
          <Marker 
            key={station.id}
            position={[station.latitude, station.longitude]}
          >
            <Popup>
              <div>
                <h3>{station.name}</h3>
                <p>Type: {station.type}</p>
                <p>Status: {station.status}</p>
                <p>Cost: ${station.cost.toFixed(2)}/hour</p>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
};

export default StationMap;