import sqlite3
import json
import os
import numpy as np
import contextlib
from typing import List, Dict, Any, Tuple

class FleetStructuralMemory:
    """
    Lightweight SQLite-based memory for fleet-wide structural events.
    Stores revisions, precursor states, alarm histories, and uncertainty.
    Fully offline, zero external dependencies.
    """
    def __init__(self, db_path: str = "fleet_live_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS engine_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    engine_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS structural_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    cycle INTEGER,
                    event_type TEXT, -- 'alarm', 'surgery_accept', 'surgery_reject'
                    edge_u INTEGER,
                    edge_v INTEGER,
                    edit_type TEXT,
                    FOREIGN KEY(run_id) REFERENCES engine_runs(run_id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS precursors (
                    event_id INTEGER,
                    leakage_hist TEXT,
                    energy_hist TEXT,
                    uncertainty_hist TEXT,
                    FOREIGN KEY(event_id) REFERENCES structural_events(event_id)
                )
            ''')
            conn.commit()

    def start_engine_run(self, engine_id: str) -> int:
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO engine_runs (engine_id) VALUES (?)", (str(engine_id),))
            conn.commit()
            return cursor.lastrowid

    def store_event(self, 
                    run_id: int, 
                    cycle: int, 
                    event_type: str, 
                    edge: Tuple[int, int], 
                    edit_type: str,
                    leakage_hist: List[float],
                    energy_hist: List[float],
                    uncertainty_hist: List[float]):
        """Compactly index fleet event with its thermodynamic precursor signature."""
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO structural_events (run_id, cycle, event_type, edge_u, edge_v, edit_type)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (run_id, cycle, event_type, edge[0], edge[1], edit_type))
            
            event_id = cursor.lastrowid
            
            cursor.execute('''
                INSERT INTO precursors (event_id, leakage_hist, energy_hist, uncertainty_hist)
                VALUES (?, ?, ?, ?)
            ''', (event_id, json.dumps(leakage_hist), json.dumps(energy_hist), json.dumps(uncertainty_hist)))
            conn.commit()

    def get_all_accepted_surgeries(self) -> List[Dict[str, Any]]:
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT e.event_id, e.run_id, r.engine_id, e.cycle, e.edge_u, e.edge_v, e.edit_type,
                       p.leakage_hist, p.energy_hist, p.uncertainty_hist
                FROM structural_events e
                JOIN engine_runs r ON e.run_id = r.run_id
                JOIN precursors p ON e.event_id = p.event_id
                WHERE e.event_type = 'surgery_accept'
            ''')
            rows = cursor.fetchall()
            
        results = []
        for row in rows:
            results.append({
                'event_id': row[0],
                'run_id': row[1],
                'engine_id': row[2],
                'cycle': row[3],
                'edge': (row[4], row[5]),
                'edit_type': row[6],
                'leakage_hist': json.loads(row[7]),
                'energy_hist': json.loads(row[8]),
                'uncertainty_hist': json.loads(row[9])
            })
        return results
