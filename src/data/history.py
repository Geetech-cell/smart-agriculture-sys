"""
Prediction history storage and management
SQLite-based storage for tracking predictions over time
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

class PredictionHistory:
    """Manage prediction history in SQLite database"""
    
    def __init__(self, db_path: str = "data/predictions.db"):
        """Initialize database connection"""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                region TEXT,
                crop TEXT,
                season TEXT,
                inputs TEXT,
                prediction REAL,
                confidence TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_prediction(self, data: Dict[str, Any]):
        """Save a new prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, region, crop, season, inputs, prediction, confidence, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            data.get('region'),
            data.get('crop'),
            data.get('season'),
            json.dumps(data.get('inputs', {})),
            data.get('prediction'),
            data.get('confidence', 'High'),
            data.get('notes', '')
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent(self, limit: int = 10) -> List[Dict]:
        """Get recent predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            if result.get('inputs'):
                result['inputs'] = json.loads(result['inputs'])
            results.append(result)
        
        conn.close()
        return results
    
    def get_by_crop(self, crop: str, limit: int = 20) -> List[Dict]:
        """Get predictions for a specific crop"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE crop = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (crop, limit))
        
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            if result.get('inputs'):
                result['inputs'] = json.loads(result['inputs'])
            results.append(result)
        
        conn.close()
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total = cursor.fetchone()[0]
        
        # Average yield
        cursor.execute('SELECT AVG(prediction) FROM predictions')
        avg_yield = cursor.fetchone()[0] or 0
        
        # By crop
        cursor.execute('''
            SELECT crop, COUNT(*), AVG(prediction) 
            FROM predictions 
            GROUP BY crop
        ''')
        by_crop = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_predictions': total,
            'avg_yield': round(avg_yield, 2),
            'by_crop': {row[0]: {'count': row[1], 'avg_yield': round(row[2], 2)} 
                       for row in by_crop if row[0]}
        }
    
    def clear_all(self):
        """Clear all prediction history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions')
        conn.commit()
        conn.close()
