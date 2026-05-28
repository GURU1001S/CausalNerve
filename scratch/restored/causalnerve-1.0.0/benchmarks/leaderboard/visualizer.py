from typing import Dict, Any, List
import numpy as np

class BenchmarkVisualizer:
    """Generates SVGs for benchmark plots."""
    
    @staticmethod
    def _create_bar_chart(title: str, labels: List[str], values: List[float], errors: List[float], color: str = "#3B82F6") -> str:
        if not labels:
            return "<svg></svg>"
            
        width, height = 400, 250
        max_val = max([v + e for v, e in zip(values, errors)]) * 1.2 if values else 1.0
        if max_val == 0: max_val = 1.0
        
        svg = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        svg += f'<text x="{width/2}" y="20" font-family="sans-serif" font-size="14" fill="#E2E8F0" text-anchor="middle" font-weight="bold">{title}</text>'
        
        n = len(labels)
        bar_width = (width - 100) / max(n, 1)
        padding = bar_width * 0.2
        active_width = bar_width - 2*padding
        
        for i, (label, val, err) in enumerate(zip(labels, values, errors)):
            x = 50 + i * bar_width + padding
            bar_h = (val / max_val) * (height - 60)
            y = height - 30 - bar_h
            
            # Bar
            svg += f'<rect x="{x}" y="{y}" width="{active_width}" height="{bar_h}" fill="{color}" rx="2" opacity="0.8" />'
            
            # Error line
            err_h = (err / max_val) * (height - 60)
            ey1 = y - err_h
            ey2 = y + err_h
            svg += f'<line x1="{x+active_width/2}" y1="{ey1}" x2="{x+active_width/2}" y2="{ey2}" stroke="#F87171" stroke-width="2" />'
            svg += f'<line x1="{x+active_width/2-3}" y1="{ey1}" x2="{x+active_width/2+3}" y2="{ey1}" stroke="#F87171" stroke-width="2" />'
            svg += f'<line x1="{x+active_width/2-3}" y2="{ey2}" x2="{x+active_width/2+3}" y2="{ey2}" stroke="#F87171" stroke-width="2" />'
            
            # Label
            svg += f'<text x="{x+active_width/2}" y="{height-10}" font-family="sans-serif" font-size="10" fill="#9CA3AF" text-anchor="middle">{label}</text>'
            # Value
            svg += f'<text x="{x+active_width/2}" y="{y-10-err_h}" font-family="sans-serif" font-size="10" fill="#E2E8F0" text-anchor="middle">{val:.2f}</text>'
            
        svg += '</svg>'
        return svg

    @staticmethod
    def generate_shd_plot(aggregated_results: Dict[str, Any]) -> str:
        labels, values, errors = [], [], []
        for model, metrics in aggregated_results.items():
            if "SHD" in metrics:
                labels.append(model)
                values.append(metrics["SHD"]["mean"])
                errors.append(metrics["SHD"]["ci_upper"] - metrics["SHD"]["mean"])
        return BenchmarkVisualizer._create_bar_chart("SHD Distribution (Lower is Better)", labels, values, errors, color="#10B981")
        
    @staticmethod
    def generate_runtime_plot(aggregated_results: Dict[str, Any]) -> str:
        labels, values, errors = [], [], []
        for model, metrics in aggregated_results.items():
            if "Runtime_sec" in metrics:
                labels.append(model)
                values.append(metrics["Runtime_sec"]["mean"])
                errors.append(metrics["Runtime_sec"]["ci_upper"] - metrics["Runtime_sec"]["mean"])
        return BenchmarkVisualizer._create_bar_chart("Runtime Scaling (Seconds)", labels, values, errors, color="#F59E0B")
