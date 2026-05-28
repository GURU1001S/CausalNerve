# Import Dependency Graph

## Dependency Direction
`causalnerve.core` <- `causalnerve.adaptation` <- `causalnerve.reasoning` <- `causalnerve_observe`

## Circular Dependency Resolution
No major circular dependencies detected.
Strict boundary enforced: `causalnerve_observe` (UI) relies on `causalnerve` (Math), but `causalnerve` does NOT import `causalnerve_observe`.
