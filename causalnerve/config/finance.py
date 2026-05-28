"""
causalnerve.config.finance
===========================
Financial market causal structure preset.
"""

from .base import CausalPreset

class FinancePreset(CausalPreset):
    """
    Financial market causal structure preset.
    Use case: model causal relationships between financial indicators,
    detect structural breaks (regime changes), simulate policy interventions.
    """
    
    name = "finance"
    n_nodes = 8
    default_persistence = 0.60    # moderate — markets change fast
    alarm_threshold = 0.10
    
    node_labels = {
        0: "CentralBankRate",
        1: "BondYield",
        2: "CreditSpread",
        3: "EquityIndex",
        4: "VIX",
        5: "FXRate",
        6: "CommodityPrice",
        7: "LiquidityConditions"
    }
