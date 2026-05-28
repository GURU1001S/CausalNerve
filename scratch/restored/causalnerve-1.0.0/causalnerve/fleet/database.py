"""
causalnerve.fleet.database
==========================
SQLite-backed store of revision events across all assets.
Domain-agnostic: stores RevisionEvent objects from any domain.
"""

import sqlite3
import json
import pandas as pd
from typing import List, Callable, Optional
from dataclasses import dataclass

from ..adapt.ocgr import RevisionEvent

class FleetRevisionDatabase:
    """
    SQLite-backed store of revision events across all assets.
    Queryable by edge, state condition, confidence, cycle.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        
    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id TEXT,
                    timestamp REAL,
                    cycle INTEGER,
                    edit_type TEXT,
                    src_node INTEGER,
                    dst_node INTEGER,
                    v_before REAL,
                    v_after REAL,
                    confidence REAL,
                    leakage_before REAL,
                    leakage_after REAL,
                    reason TEXT,
                    accepted BOOLEAN,
                    state_snapshot TEXT
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_asset ON revisions(asset_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edge ON revisions(src_node, dst_node)")

    def ingest(self, event: RevisionEvent, asset_id: str, state_snapshot: Optional[List[float]] = None):
        """Add a revision event from any asset."""
        src, dst = event.edge
        state_str = json.dumps(state_snapshot) if state_snapshot is not None else "{}"
        
        with self.conn:
            self.conn.execute("""
                INSERT INTO revisions (
                    asset_id, timestamp, cycle, edit_type, src_node, dst_node,
                    v_before, v_after, confidence, leakage_before, leakage_after,
                    reason, accepted, state_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                asset_id, event.timestamp, event.cycle, event.edit_type, src, dst,
                event.V_before, event.V_after, event.confidence, 
                event.leakage_before, event.leakage_after,
                event.reason, event.accepted, state_str
            ))

    def _row_to_event(self, row) -> RevisionEvent:
        return RevisionEvent(
            timestamp=row[2],
            cycle=row[3],
            edit_type=row[4],
            edge=(row[5], row[6]),
            V_before=row[7],
            V_after=row[8],
            confidence=row[9],
            leakage_before=row[10],
            leakage_after=row[11],
            reason=row[12],
            accepted=bool(row[13])
        )

    def query_by_edge(self, 
                      src: int, dst: int,
                      accepted_only: bool = True,
                      min_confidence: float = 0.0) -> List[RevisionEvent]:
        """All events for a specific edge across all assets."""
        query = "SELECT * FROM revisions WHERE src_node = ? AND dst_node = ? AND confidence >= ?"
        params = [src, dst, min_confidence]
        
        if accepted_only:
            query += " AND accepted = 1"
            
        cursor = self.conn.execute(query, params)
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def query_by_state_condition(self,
                                 state_fn: Callable[[List[float]], bool],
                                 accepted_only: bool = True) -> List[RevisionEvent]:
        """
        Find events where state_fn(state_snapshot) == True.
        """
        query = "SELECT * FROM revisions"
        if accepted_only:
            query += " WHERE accepted = 1"
            
        cursor = self.conn.execute(query)
        results = []
        for row in cursor.fetchall():
            state_str = row[14]
            if state_str != "{}":
                try:
                    state = json.loads(state_str)
                    if state_fn(state):
                        results.append(self._row_to_event(row))
                except json.JSONDecodeError:
                    pass
        return results

    def recurrence_map(self) -> pd.DataFrame:
        """
        Summary table: which edges recur, in how many assets,
        under what conditions, at what confidence.
        """
        query = """
            SELECT src_node, dst_node, edit_type, 
                   COUNT(DISTINCT asset_id) as asset_count,
                   COUNT(id) as total_occurrences,
                   AVG(confidence) as mean_confidence
            FROM revisions
            WHERE accepted = 1
            GROUP BY src_node, dst_node, edit_type
            ORDER BY asset_count DESC, total_occurrences DESC
        """
        return pd.read_sql_query(query, self.conn)
