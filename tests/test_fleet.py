"""
Tests for causalnerve.fleet
"""

import pytest
from causalnerve.fleet import FleetRevisionDatabase

def test_database_empty_query():
    db = FleetRevisionDatabase()
    assert len(db.query_by_edge(0, 1)) == 0
