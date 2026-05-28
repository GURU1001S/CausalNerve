# Render Pipeline

1. **State Manager:** Mutates in memory.
2. **Diff Engine:** Calculates precise structural edge/node topology changes.
3. **Event Bus:** Dispatches targeted events to active Tab context only.
4. **Frame Throttler:** Batches updates to maintain 60 FPS limits via asyncio.
5. **WebGL Canvas:** go.Scattergl consumes the JSON payload for hardware-accelerated drawing.
