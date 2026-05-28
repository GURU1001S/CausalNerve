# Plugin Matrix

| Plugin Domain | Capabilities | Sensors | Configs |
|---|---|---|---|
| Aerospace (Turbofan) | Degradation, RUL | 24 | `preset_turbofan` |
| Medical (ICU) | Hemodynamics, Sepsis | 16 | `preset_icu` |
| Industrial (Wind) | SCADA, Vibration | 14 | `preset_wind` |

All plugins now adhere to strict domain-agnostic `CausalNerve` initializations. Domain isolation is maintained.
