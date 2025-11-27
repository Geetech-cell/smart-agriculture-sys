"""
Weather API Integration
Simple weather data fetching (works without API key for demo)
"""
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import random

class WeatherAPI:
    """Weather data provider with fallback to mock data"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with optional API key"""
        self.api_key = api_key
        self.base_url = "https://api.open-meteo.com/v1/forecast"  # Free, no key needed
    
    def get_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get current weather and forecast
        Uses open-meteo.com (free, no API key required)
        """
        try:
            # Try real API (open-meteo is free and doesn't need a key)
            params = {
                'latitude': lat,
                'longitude': lon,
                'current_weather': True,
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum',
                'timezone': 'auto',
                'forecast_days': 7
            }
            
            response = requests.get(self.base_url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return self._format_weather_data(data)
            else:
                return self._get_mock_weather(lat, lon)
                
        except Exception as e:
            print(f"Weather API error: {e}")
            return self._get_mock_weather(lat, lon)
    
    def _format_weather_data(self, data: Dict) -> Dict[str, Any]:
        """Format API response"""
        current = data.get('current_weather', {})
        daily = data.get('daily', {})
        
        return {
            'current_temp': current.get('temperature', 25),
            'wind_speed': current.get('windspeed', 10),
            'weather_code': current.get('weathercode', 0),
            'forecast': {
                'dates': daily.get('time', []),
                'temp_max': daily.get('temperature_2m_max', []),
                'temp_min': daily.get('temperature_2m_min', []),
                'precipitation': daily.get('precipitation_sum', [])
            },
            'source': 'Open-Meteo API',
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_mock_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """Generate realistic mock weather data"""
        base_temp = 20 + (abs(lat) / 90) * 15  # Temperature based on latitude
        
        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range(7)]
        
        return {
            'current_temp': round(base_temp + random.uniform(-5, 5), 1),
            'wind_speed': round(random.uniform(5, 20), 1),
            'weather_code': random.choice([0, 1, 2, 3]),
            'forecast': {
                'dates': dates,
                'temp_max': [round(base_temp + random.uniform(0, 8), 1) for _ in range(7)],
                'temp_min': [round(base_temp - random.uniform(0, 8), 1) for _ in range(7)],
                'precipitation': [round(random.uniform(0, 20), 1) for _ in range(7)]
            },
            'source': 'Mock Data (for demonstration)',
            'timestamp': datetime.now().isoformat()
        }

# Region coordinates for weather lookup
REGION_COORDS = {
    "Northern": {"lat": 9.5, "lon": -12.0},
    "Southern": {"lat": -15.3, "lon": 28.3},
    "Eastern": {"lat": -1.3, "lon": 36.8},
    "Western": {"lat": 6.5, "lon": -0.2},
    "Central": {"lat": 0.3, "lon": 32.6},
    "Highland": {"lat": -13.3, "lon": 34.0},
}

def get_weather_for_region(region: str) -> Dict[str, Any]:
    """Get weather data for a specific region"""
    coords = REGION_COORDS.get(region, {"lat": 0, "lon": 0})
    weather_api = WeatherAPI()
    return weather_api.get_weather(coords['lat'], coords['lon'])
