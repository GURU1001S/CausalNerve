from typing import Dict, Any, List
from benchmarks.leaderboard.aggregator import ResultsAggregator
from benchmarks.leaderboard.visualizer import BenchmarkVisualizer
from datetime import datetime

class LeaderboardReportGenerator:
    """Generates the interactive HTML Leaderboard Report."""
    
    @staticmethod
    def generate_html(raw_results: Dict[str, Dict[str, List[float]]], failures: List[Dict[str, Any]], filepath: str = "LEADERBOARD.html"):
        aggregated = ResultsAggregator.aggregate(raw_results)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CausalNerve Leaderboard</title>
<style>
    body {{ font-family: 'Inter', system-ui, sans-serif; background: #0F172A; color: #E2E8F0; margin: 0; padding: 40px; }}
    h1 {{ color: #F8FAFC; border-bottom: 1px solid #334155; padding-bottom: 10px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #1E293B; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #334155; }}
    th {{ background: #0F172A; color: #94A3B8; text-transform: uppercase; font-size: 12px; cursor: pointer; }}
    th:hover {{ color: #F8FAFC; }}
    tr:hover {{ background: #334155; }}
    .charts {{ display: flex; gap: 20px; margin-top: 40px; flex-wrap: wrap; }}
    .chart-card {{ background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 20px; }}
    .failures {{ margin-top: 40px; background: rgba(220, 38, 38, 0.1); border: 1px solid #DC2626; border-radius: 8px; padding: 20px; }}
    .failures h2 {{ color: #FCA5A5; margin-top: 0; }}
</style>
</head>
<body>
    <h1>CausalNerve Scientific Leaderboard</h1>
    <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    
    <table id="leaderboard-table">
        <thead>
            <tr>
                <th onclick="sortTable(0)">Model ↕</th>
                <th onclick="sortTable(1)">SHD ↕</th>
                <th onclick="sortTable(2)">SID ↕</th>
                <th onclick="sortTable(3)">ECE ↕</th>
                <th onclick="sortTable(4)">False Surgery Rate ↕</th>
                <th onclick="sortTable(5)">Runtime (s) ↕</th>
            </tr>
        </thead>
        <tbody>
"""
        # Table rows
        for model_name, metrics in aggregated.items():
            shd = f"{metrics['SHD']['mean']:.2f} ± {metrics['SHD']['ci_upper'] - metrics['SHD']['mean']:.2f}" if "SHD" in metrics else "N/A"
            sid = f"{metrics['SID']['mean']:.2f} ± {metrics['SID']['ci_upper'] - metrics['SID']['mean']:.2f}" if "SID" in metrics else "N/A"
            ece = f"{metrics['ECE']['mean']:.3f}" if "ECE" in metrics else "N/A"
            fsr = f"{metrics['False_Surgery_Rate']['mean']:.3f}" if "False_Surgery_Rate" in metrics else "N/A"
            rt = f"{metrics['Runtime_sec']['mean']:.3f}" if "Runtime_sec" in metrics else "N/A"
            
            html += f"<tr><td><strong>{model_name}</strong></td><td>{shd}</td><td>{sid}</td><td>{ece}</td><td>{fsr}</td><td>{rt}</td></tr>"
            
        html += """
        </tbody>
    </table>
    
    <div class="charts">
"""
        html += f'<div class="chart-card">{BenchmarkVisualizer.generate_shd_plot(aggregated)}</div>'
        html += f'<div class="chart-card">{BenchmarkVisualizer.generate_runtime_plot(aggregated)}</div>'
        
        html += """
    </div>
"""
        if failures:
            html += """<div class="failures"><h2>Failure Boundary Report</h2><ul>"""
            for f in failures:
                html += f"<li><strong>{f['model']}</strong> (Seed {f['seed']}): {f['error']}</li>"
            html += "</ul></div>"

        html += """
    <script>
    function sortTable(n) {
      var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
      table = document.getElementById("leaderboard-table");
      switching = true;
      dir = "asc"; 
      while (switching) {
        switching = false;
        rows = table.rows;
        for (i = 1; i < (rows.length - 1); i++) {
          shouldSwitch = false;
          x = rows[i].getElementsByTagName("TD")[n];
          y = rows[i + 1].getElementsByTagName("TD")[n];
          var xv = parseFloat(x.innerHTML.split(' ')[0]) || x.innerHTML.toLowerCase();
          var yv = parseFloat(y.innerHTML.split(' ')[0]) || y.innerHTML.toLowerCase();
          if (dir == "asc") {
            if (xv > yv) { shouldSwitch = true; break; }
          } else if (dir == "desc") {
            if (xv < yv) { shouldSwitch = true; break; }
          }
        }
        if (shouldSwitch) {
          rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
          switching = true;
          switchcount ++;      
        } else {
          if (switchcount == 0 && dir == "asc") {
            dir = "desc";
            switching = true;
          }
        }
      }
    }
    </script>
</body>
</html>
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"[Leaderboard] Generated interactive HTML report: {filepath}")
