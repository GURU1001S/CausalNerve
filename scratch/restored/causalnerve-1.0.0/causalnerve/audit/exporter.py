import json
import datetime
from typing import List, Dict, Any
from causalnerve.reporting.trail import AuditTrail

class AuditExporter:
    """Exports scientific audit trails to various formats."""
    
    @staticmethod
    def export_json(trail: AuditTrail, filepath: str = "audit_report.json"):
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "audit_trail": trail.get_full_audit()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    @staticmethod
    def export_markdown(trail: AuditTrail, filepath: str = "audit_report.md"):
        lines = []
        lines.append("# CausalNerve Scientific Audit Report")
        lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n## Intervention History\n")
        
        for node in trail.history:
            lines.append(f"### Event: {node.event_id} ({node.action})")
            lines.append(f"- **Edge**: {node.edge}")
            lines.append(f"- **Reasoning**: {node.reasoning}")
            lines.append("- **Evidence**:")
            ev = node.evidence
            lines.append(f"  - Leakage Before: {ev.leakage_before:.4f}")
            lines.append(f"  - Leakage After: {ev.leakage_after:.4f}")
            lines.append(f"  - Confidence: {ev.confidence_score:.2f}")
            lines.append(f"  - Counterfactual Divergence: {ev.counterfactual_divergence:.3f}")
            if ev.physical_constraints_triggered:
                lines.append(f"  - Constraints Triggered: {', '.join(ev.physical_constraints_triggered)}")
            if node.dependencies:
                lines.append(f"- **Dependencies**: {', '.join(node.dependencies)}")
            lines.append("\n---")
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
            
    @staticmethod
    def export_pdf(trail: AuditTrail, filepath: str = "audit_report.pdf"):
        """Stub for PDF generation (could use reportlab or weasyprint)."""
        print(f"PDF export not implemented. Please use markdown export: {filepath.replace('.pdf', '.md')}")
