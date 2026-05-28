"""
causalnerve.visualization_stub.theme
=====================
CausalNerve visual language — consistent across all visualizations.
"""

COLORS = {
    "node_healthy":    "#1D9E75",    # green — normal
    "node_warning":    "#EF9F27",    # amber — precursor
    "node_alarm":      "#E24B4A",    # red — alarm
    "node_intervened": "#7F77DD",    # purple — under do()
    "edge_active":     "#378ADD",    # blue — carrying signal
    "edge_severed":    "#B4B2A9",    # gray — severed
    "edge_new":        "#9FE1CB",    # teal — newly added
    "edge_removed":    "#FAC775",    # amber — recently removed
    "background":      "#FAFAF8",
    "text":            "#1A1916",
}
