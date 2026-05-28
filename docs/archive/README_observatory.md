# CausalNerve Intelligence Observatory

**NASA C-MAPSS FD004 · Autonomous Causal Monitoring Dashboard**

## Quick Start (Demo — no model required)

```bash
pip install fastapi uvicorn numpy
python causalnerve_demo.py
```

Dashboard opens automatically at `http://localhost:8765`.

## Integration (3 lines)

```python
from causalnerve_observatory import CausalNerveObservatory

obs = CausalNerveObservatory(port=8765, scenario="fd004", auto_open=True)
obs.start()

# In your training / monitoring loop:
obs.update(cycle, {
    "edge_probs": edge_probs,        # (14,14) ndarray or tensor
    "loss": loss,                     # float
    "leakage_L": leakage_L,          # float
    "lyapunov_V": lyapunov_V,        # float
    "sensor_values": sensor_values,  # (21,) sensor readings
    # ... see causalnerve_observatory.py for full schema
})
```

## Dependencies

| Package  | Purpose            |
|----------|--------------------|
| fastapi  | API server         |
| uvicorn  | ASGI runner        |
| numpy    | Numerical backend  |

## Files

| File                        | Description                         |
|-----------------------------|-------------------------------------|
| `causalnerve_observatory.py`| Self-contained server + embedded UI |
| `causalnerve_demo.py`       | Standalone demo (simulated FD004)   |
| `README_observatory.md`     | This file                           |

## API Endpoints

| Route               | Description                        |
|----------------------|------------------------------------|
| `GET /`              | Live dashboard                     |
| `GET /api/state`     | Current state JSON                 |
| `GET /api/history`   | Last N metric snapshots            |
| `GET /api/audit`     | Audit log (JSON)                   |
| `GET /api/audit.ndjson` | Audit log (NDJSON download)     |

## License

Same as CausalNerve (MIT).
