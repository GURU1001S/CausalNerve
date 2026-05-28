import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os

def create_flagship_gif():
    # --- CONFIGURATION ---
    # Colors (GitHub Dark modern theme inspired)
    BG_COLOR = "#0d1117"
    TEXT_COLOR = "#c9d1d9"
    NODE_HEALTHY = "#2ea043"      # Green
    NODE_ALARM = "#f85149"        # Red
    EDGE_ACTIVE = "#58a6ff"       # Blue
    EDGE_HIDDEN = "#d29922"       # Orange/Yellow
    EDGE_NEW = "#3fb950"          # Teal
    PANEL_BG = "#161b22"
    
    # 60 frames = 30 seconds @ 2fps
    FPS = 2
    FRAMES = 60
    
    # Fixed Circular Layout for 6 nodes
    nodes = ["Fan", "HPC", "LPT", "HPT", "Burner", "Shaft"]
    n_nodes = 6
    theta = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    pos = {i: (np.cos(t), np.sin(t)) for i, t in enumerate(theta)}
    
    # Base Graph
    base_edges = [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5)]
    
    # --- SETUP FIGURE ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 7), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(3, 4, figure=fig)
    
    # Main graph area (span 3 rows, 3 cols)
    ax_graph = fig.add_subplot(gs[0:3, 0:3])
    ax_graph.set_facecolor(BG_COLOR)
    ax_graph.axis('off')
    
    # Side panel for leakage (span 1 row, 1 col)
    ax_leakage = fig.add_subplot(gs[0, 3])
    ax_leakage.set_facecolor(PANEL_BG)
    ax_leakage.set_title("Leakage Monitor", color=TEXT_COLOR, fontsize=10)
    
    # Side panel for V(G) Energy (span 1 row, 1 col)
    ax_energy = fig.add_subplot(gs[1, 3])
    ax_energy.set_facecolor(PANEL_BG)
    ax_energy.set_title("Structural Energy V(G)", color=TEXT_COLOR, fontsize=10)
    
    # Side panel for Audit Trail (span 1 row, 1 col)
    ax_audit = fig.add_subplot(gs[2, 3])
    ax_audit.set_facecolor(PANEL_BG)
    ax_audit.set_title("Revision Log", color=TEXT_COLOR, fontsize=10)
    ax_audit.axis('off')

    # Data for plots
    leakage_hist = []
    energy_hist = []
    
    # Titles
    title_text = fig.suptitle("", fontsize=18, color=TEXT_COLOR, y=0.95, weight='bold')
    
    # --- DRAWING FUNCTIONS ---
    def draw_nodes(ax, colors):
        nx.draw_networkx_nodes(G=nx.DiGraph(), pos=pos, nodelist=range(6), 
                               ax=ax, node_color=colors, node_size=2000, 
                               edgecolors="white", linewidths=2)
        nx.draw_networkx_labels(G=nx.DiGraph(), pos=pos, 
                                labels={i: nodes[i] for i in range(6)}, 
                                font_size=11, font_color="white", font_weight='bold', ax=ax)
                                
    def draw_edges(ax, edges, color, style='solid', width=2):
        G = nx.DiGraph()
        G.add_edges_from(edges)
        nx.draw_networkx_edges(G, pos=pos, ax=ax, edgelist=edges, 
                               edge_color=color, style=style, width=width, 
                               arrowsize=20, node_size=2000, connectionstyle="arc3,rad=0.1")

    # --- ANIMATION FUNCTION ---
    def update(frame):
        ax_graph.clear()
        ax_graph.axis('off')
        ax_leakage.clear()
        ax_energy.clear()
        ax_audit.clear()
        ax_audit.axis('off')
        
        ax_leakage.set_facecolor(PANEL_BG)
        ax_energy.set_facecolor(PANEL_BG)
        ax_leakage.set_title("Leakage Monitor", color=TEXT_COLOR, fontsize=10)
        ax_energy.set_title("Structural Energy V(G)", color=TEXT_COLOR, fontsize=10)
        ax_audit.set_title("Revision Log", color=TEXT_COLOR, fontsize=10)
        
        # Default states
        node_colors = [NODE_HEALTHY] * 6
        current_edges = list(base_edges)
        
        # Frame Logic
        if frame < 10:
            # STABLE STATE
            title = "Stable causal structure"
            leak = np.random.uniform(0.01, 0.03)
            energy = 4.2 + np.random.uniform(-0.1, 0.1)
            draw_edges(ax_graph, current_edges, EDGE_ACTIVE)
            draw_nodes(ax_graph, node_colors)
            
        elif frame < 15:
            # HIDDEN DRIFT
            title = "Unseen structural change..."
            leak = 0.02 + ((frame - 10) / 5) * 0.15 # Rising slowly
            energy = 4.2 + ((frame - 10) / 5) * 1.5
            draw_edges(ax_graph, current_edges, EDGE_ACTIVE)
            draw_edges(ax_graph, [(3, 1)], EDGE_HIDDEN, style='dashed')
            draw_nodes(ax_graph, node_colors)
            ax_graph.text(0, -1.3, "Hidden coupling emerges: HPT → HPC", 
                          color=EDGE_HIDDEN, fontsize=12, ha='center', weight='bold')
            
        elif frame < 22:
            # ALARM FIRES
            title = "Causal alarm detected"
            leak = 0.17 + ((frame - 15) / 7) * 0.4 # Sharp spike
            energy = 5.7 + ((frame - 15) / 7) * 2.0
            
            # Pulse effect
            if frame % 2 == 0:
                node_colors[1] = NODE_ALARM
                node_colors[3] = NODE_ALARM
                
            draw_edges(ax_graph, current_edges, EDGE_ACTIVE)
            draw_edges(ax_graph, [(3, 1)], NODE_ALARM, style='dashed', width=3)
            draw_nodes(ax_graph, node_colors)
            
            # Badge
            rect = patches.Rectangle((-0.5, 0.0), 1.0, 0.2, linewidth=1, 
                                     edgecolor=NODE_ALARM, facecolor=PANEL_BG, zorder=0)
            ax_graph.add_patch(rect)
            ax_graph.text(0, 0.1, "Structural Alarm: 4→2", color=NODE_ALARM, 
                          ha='center', va='center', weight='bold')
            
        elif frame < 28:
            # COUNTERFACTUAL WORLDS
            title = "Testing counterfactual: would this edit help?"
            leak = 0.57 # Flat in reality, we show divergence in plot
            energy = 7.7
            
            # Draw split screen concept via drawing two small graphs
            ax_graph.clear()
            ax_graph.axis('off')
            
            # World 0 (Left)
            pos0 = {i: (p[0]*0.4 - 0.5, p[1]*0.4) for i, p in pos.items()}
            G0 = nx.DiGraph()
            G0.add_edges_from(current_edges)
            nx.draw_networkx_edges(G0, pos=pos0, ax=ax_graph, edge_color=EDGE_ACTIVE, node_size=500, width=1.5, connectionstyle="arc3,rad=0.1")
            nx.draw_networkx_nodes(G0, pos=pos0, ax=ax_graph, node_color=[NODE_ALARM]*6, node_size=500)
            ax_graph.text(-0.5, -0.6, "World 0 (No Edit)\nLeakage Rising", color=NODE_ALARM, ha='center', fontsize=10)
            
            # World 1 (Right)
            pos1 = {i: (p[0]*0.4 + 0.5, p[1]*0.4) for i, p in pos.items()}
            G1 = nx.DiGraph()
            G1.add_edges_from(current_edges + [(3,1)])
            nx.draw_networkx_edges(G1, pos=pos1, ax=ax_graph, edge_color=EDGE_ACTIVE, node_size=500, width=1.5, connectionstyle="arc3,rad=0.1")
            nx.draw_networkx_edges(G1, pos=pos1, edgelist=[(3,1)], ax=ax_graph, edge_color=EDGE_NEW, node_size=500, width=2.5, connectionstyle="arc3,rad=0.1")
            nx.draw_networkx_nodes(G1, pos=pos1, ax=ax_graph, node_color=[NODE_HEALTHY]*6, node_size=500)
            ax_graph.text(0.5, -0.6, "World 1 (Edit 4→2)\nLeakage Dropping", color=EDGE_NEW, ha='center', fontsize=10)
            
        elif frame < 34:
            # LYAPUNOV VALIDATION
            title = "Stability check passed"
            leak = 0.57
            energy = 7.7
            
            draw_edges(ax_graph, current_edges, EDGE_ACTIVE)
            draw_edges(ax_graph, [(3, 1)], EDGE_NEW, style='dashed', width=3)
            draw_nodes(ax_graph, node_colors)
            
            # Validation Card
            rect = patches.Rectangle((-0.4, -0.2), 0.8, 0.4, linewidth=2, 
                                     edgecolor=EDGE_NEW, facecolor=PANEL_BG, zorder=0)
            ax_graph.add_patch(rect)
            ax_graph.text(0, 0.1, "Energy decreases ✓", color=EDGE_NEW, ha='center', fontsize=12, weight='bold')
            ax_graph.text(0, -0.1, "Confidence: 0.82", color=EDGE_ACTIVE, ha='center', fontsize=12, weight='bold')
            
        elif frame < 42:
            # GRAPH SURGERY
            title = "Graph repaired — no retraining needed"
            leak = max(0.05, 0.57 - ((frame - 34) / 8) * 0.52)
            energy = max(3.5, 7.7 - ((frame - 34) / 8) * 4.2)
            current_edges.append((3, 1))
            
            draw_edges(ax_graph, base_edges, EDGE_ACTIVE)
            draw_edges(ax_graph, [(3, 1)], EDGE_NEW, width=3)
            draw_nodes(ax_graph, node_colors)
            
            ax_graph.text(0, -1.3, "Added edge 4→2: thermal feedback coupling detected", 
                          color=EDGE_NEW, fontsize=11, ha='center', weight='bold')
            
        elif frame < 50:
            # STABLE AGAIN
            title = "Causal equilibrium restored"
            leak = np.random.uniform(0.02, 0.05)
            energy = 3.5 + np.random.uniform(-0.1, 0.1)
            current_edges.append((3, 1))
            
            draw_edges(ax_graph, current_edges, EDGE_ACTIVE)
            draw_nodes(ax_graph, node_colors)
            
        elif frame < 55:
            # AUDIT TRAIL
            title = "Every change is traceable"
            leak = 0.03
            energy = 3.5
            current_edges.append((3, 1))
            draw_edges(ax_graph, current_edges, EDGE_ACTIVE)
            draw_nodes(ax_graph, node_colors)
            
        else:
            # OUTRO
            title = "Living causal graphs for any domain"
            leak = 0.03
            energy = 3.5
            ax_graph.text(0, 0.2, "pip install causalnerve", color=TEXT_COLOR, fontsize=20, ha='center', weight='bold')
            ax_graph.text(0, -0.2, "github.com/agency/CausalNerve", color=EDGE_ACTIVE, fontsize=16, ha='center')
        
        # Update history
        leakage_hist.append(leak)
        energy_hist.append(energy)
        
        # Plot Leakage
        ax_leakage.plot(leakage_hist, color=NODE_ALARM if leak > 0.15 else EDGE_NEW, linewidth=2)
        ax_leakage.set_xlim(0, 60)
        ax_leakage.set_ylim(0, 0.7)
        ax_leakage.tick_params(colors=TEXT_COLOR, labelsize=8)
        
        # Plot Energy
        ax_energy.plot(energy_hist, color=EDGE_ACTIVE, linewidth=2)
        ax_energy.set_xlim(0, 60)
        ax_energy.set_ylim(2, 9)
        ax_energy.tick_params(colors=TEXT_COLOR, labelsize=8)
        
        # Plot Audit
        if frame >= 35:
            ax_audit.text(0.05, 0.8, "Revision 1042", color=TEXT_COLOR, fontsize=9, weight='bold')
            ax_audit.text(0.05, 0.6, "Time: Cycle 42,912", color=TEXT_COLOR, fontsize=8)
            ax_audit.text(0.05, 0.4, "Action: ADD 4→2", color=EDGE_NEW, fontsize=9, weight='bold')
            ax_audit.text(0.05, 0.2, "Conf: 0.82 | Leak: -18%", color=TEXT_COLOR, fontsize=8)
            ax_audit.text(0.05, 0.0, "Reason: Dual-world validation passed", color=TEXT_COLOR, fontsize=8)
            
        title_text.set_text(title)

def is_ffmpeg_available() -> bool:
    return animation.writers.is_available("ffmpeg")

def print_ffmpeg_install_help():
    print("\n--- ffmpeg Installation Guide ---")
    print("Windows: winget install ffmpeg  (or download from gyan.dev)")
    print("Ubuntu:  sudo apt install ffmpeg")
    print("macOS:   brew install ffmpeg")
    print("---------------------------------\n")

def create_flagship_gif():
    fig, ax_graph, ax_leakage, ax_energy, ax_audit, title_text, pos, base_edges, update_func, FRAMES, FPS = _setup_figure()
    os.makedirs("assets", exist_ok=True)
    
    print("Generating Flagship GIF (this may take a minute)...")
    anim = animation.FuncAnimation(fig, update_func, frames=FRAMES, interval=1000/FPS)
    
    # 1. ALWAYS EXPORT GIF
    gif_path = "assets/flagship_demo.gif"
    # Optimized DPI to keep file size small while looking sharp
    anim.save(gif_path, writer='pillow', fps=FPS, dpi=120)
    gif_size = os.path.getsize(gif_path) / (1024 * 1024)
    print(f"Saved GIF to {gif_path} ({gif_size:.2f} MB)")
    gif_status = "SUCCESS"
    
    # 2. CONDITIONALLY EXPORT MP4
    mp4_path = "assets/flagship_demo.mp4"
    mp4_status = "SKIPPED"
    
    if is_ffmpeg_available():
        try:
            print("ffmpeg found. Generating MP4...")
            anim.save(mp4_path, writer='ffmpeg', fps=FPS, dpi=120, bitrate=1800)
            mp4_size = os.path.getsize(mp4_path) / (1024 * 1024)
            print(f"Saved MP4 to {mp4_path} ({mp4_size:.2f} MB)")
            mp4_status = "SUCCESS"
        except Exception as e:
            print(f"[ERROR] ffmpeg crashed during export: {e}")
            mp4_status = "FAILED"
    else:
        print("[WARN] ffmpeg not installed. MP4 export skipped.")
        print_ffmpeg_install_help()
        
    print("\n=== EXPORT SUMMARY ===")
    print(f"GIF: {gif_status}")
    print(f"MP4: {mp4_status}")

def _setup_figure():
    # Helper to encapsulate the setup logic
    BG_COLOR = "#0d1117"
    TEXT_COLOR = "#c9d1d9"
    NODE_HEALTHY = "#2ea043"      # Green
    NODE_ALARM = "#f85149"        # Red
    EDGE_ACTIVE = "#58a6ff"       # Blue
    EDGE_HIDDEN = "#d29922"       # Orange/Yellow
    EDGE_NEW = "#3fb950"          # Teal
    PANEL_BG = "#161b22"
    
    FPS = 2
    FRAMES = 60
    
    nodes = ["Fan", "HPC", "LPT", "HPT", "Burner", "Shaft"]
    n_nodes = 6
    theta = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    pos = {i: (np.cos(t), np.sin(t)) for i, t in enumerate(theta)}
    base_edges = [(0, 1), (1, 4), (4, 3), (3, 2), (2, 5)]
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 7), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(3, 4, figure=fig)
    
    ax_graph = fig.add_subplot(gs[0:3, 0:3])
    ax_graph.set_facecolor(BG_COLOR)
    ax_graph.axis('off')
    
    ax_leakage = fig.add_subplot(gs[0, 3])
    ax_leakage.set_facecolor(PANEL_BG)
    ax_leakage.set_title("Leakage Monitor", color=TEXT_COLOR, fontsize=10)
    
    ax_energy = fig.add_subplot(gs[1, 3])
    ax_energy.set_facecolor(PANEL_BG)
    ax_energy.set_title("Structural Energy V(G)", color=TEXT_COLOR, fontsize=10)
    
    ax_audit = fig.add_subplot(gs[2, 3])
    ax_audit.set_facecolor(PANEL_BG)
    ax_audit.set_title("Revision Log", color=TEXT_COLOR, fontsize=10)
    ax_audit.axis('off')

    leakage_hist = []
    energy_hist = []
    title_text = fig.suptitle("", fontsize=18, color=TEXT_COLOR, y=0.95, weight='bold')

    def draw_nodes(ax, colors):
        nx.draw_networkx_nodes(G=nx.DiGraph(), pos=pos, nodelist=range(6), 
                               ax=ax, node_color=colors, node_size=2000, 
                               edgecolors="white", linewidths=2)
        nx.draw_networkx_labels(G=nx.DiGraph(), pos=pos, 
                                labels={i: nodes[i] for i in range(6)}, 
                                font_size=11, font_color="white", font_weight='bold', ax=ax)
                                
    def draw_edges(ax, edges, color, style='solid', width=2):
        G = nx.DiGraph()
        G.add_edges_from(edges)
        nx.draw_networkx_edges(G, pos=pos, ax=ax, edgelist=edges, 
                               edge_color=color, style=style, width=width, 
                               arrowsize=20, node_size=2000, connectionstyle="arc3,rad=0.1")

    def update(frame):
        ax_graph.clear()
        ax_graph.axis('off')
        ax_leakage.clear()
        ax_energy.clear()
        ax_audit.clear()
        ax_audit.axis('off')
        
        ax_leakage.set_facecolor(PANEL_BG)
        ax_energy.set_facecolor(PANEL_BG)
        ax_leakage.set_title("Leakage Monitor", color=TEXT_COLOR, fontsize=10)
        ax_energy.set_title("Structural Energy V(G)", color=TEXT_COLOR, fontsize=10)
        ax_audit.set_title("Revision Log", color=TEXT_COLOR, fontsize=10)
        
        node_colors = [NODE_HEALTHY] * 6
        current_edges = list(base_edges)
        
        if frame < 10:
            title = "Stable causal structure"
            leak = np.random.uniform(0.01, 0.03)
            energy = 4.2 + np.random.uniform(-0.1, 0.1)
            draw_edges(ax_graph, current_edges, EDGE_ACTIVE)
            draw_nodes(ax_graph, node_colors)
            
        elif frame < 15:
            title = "Unseen structural change..."
            leak = 0.02 + ((frame - 10) / 5) * 0.15
            energy = 4.2 + ((frame - 10) / 5) * 1.5
            draw_edges(ax_graph, current_edges, EDGE_ACTIVE)
            draw_edges(ax_graph, [(3, 1)], EDGE_HIDDEN, style='dashed')
            draw_nodes(ax_graph, node_colors)
            ax_graph.text(0, -1.3, "Hidden coupling emerges: HPT → HPC", 
                          color=EDGE_HIDDEN, fontsize=12, ha='center', weight='bold')
            
        elif frame < 22:
            title = "Causal alarm detected"
            leak = 0.17 + ((frame - 15) / 7) * 0.4
            energy = 5.7 + ((frame - 15) / 7) * 2.0
            if frame % 2 == 0:
                node_colors[1] = NODE_ALARM
                node_colors[3] = NODE_ALARM
                
            draw_edges(ax_graph, current_edges, EDGE_ACTIVE)
            draw_edges(ax_graph, [(3, 1)], NODE_ALARM, style='dashed', width=3)
            draw_nodes(ax_graph, node_colors)
            
            rect = patches.Rectangle((-0.5, 0.0), 1.0, 0.2, linewidth=1, 
                                     edgecolor=NODE_ALARM, facecolor=PANEL_BG, zorder=0)
            ax_graph.add_patch(rect)
            ax_graph.text(0, 0.1, "Structural Alarm: 4→2", color=NODE_ALARM, 
                          ha='center', va='center', weight='bold')
            
        elif frame < 28:
            title = "Testing counterfactual: would this edit help?"
            leak = 0.57
            energy = 7.7
            ax_graph.clear()
            ax_graph.axis('off')
            
            pos0 = {i: (p[0]*0.4 - 0.5, p[1]*0.4) for i, p in pos.items()}
            G0 = nx.DiGraph()
            G0.add_edges_from(current_edges)
            nx.draw_networkx_edges(G0, pos=pos0, ax=ax_graph, edge_color=EDGE_ACTIVE, node_size=500, width=1.5, connectionstyle="arc3,rad=0.1")
            nx.draw_networkx_nodes(G0, pos=pos0, ax=ax_graph, node_color=[NODE_ALARM]*6, node_size=500)
            ax_graph.text(-0.5, -0.6, "World 0 (No Edit)\nLeakage Rising", color=NODE_ALARM, ha='center', fontsize=10)
            
            pos1 = {i: (p[0]*0.4 + 0.5, p[1]*0.4) for i, p in pos.items()}
            G1 = nx.DiGraph()
            G1.add_edges_from(current_edges + [(3,1)])
            nx.draw_networkx_edges(G1, pos=pos1, ax=ax_graph, edge_color=EDGE_ACTIVE, node_size=500, width=1.5, connectionstyle="arc3,rad=0.1")
            nx.draw_networkx_edges(G1, pos=pos1, edgelist=[(3,1)], ax=ax_graph, edge_color=EDGE_NEW, node_size=500, width=2.5, connectionstyle="arc3,rad=0.1")
            nx.draw_networkx_nodes(G1, pos=pos1, ax=ax_graph, node_color=[NODE_HEALTHY]*6, node_size=500)
            ax_graph.text(0.5, -0.6, "World 1 (Edit 4→2)\nLeakage Dropping", color=EDGE_NEW, ha='center', fontsize=10)
            
        elif frame < 34:
            title = "Stability check passed"
            leak = 0.57
            energy = 7.7
            draw_edges(ax_graph, current_edges, EDGE_ACTIVE)
            draw_edges(ax_graph, [(3, 1)], EDGE_NEW, style='dashed', width=3)
            draw_nodes(ax_graph, node_colors)
            
            rect = patches.Rectangle((-0.4, -0.2), 0.8, 0.4, linewidth=2, 
                                     edgecolor=EDGE_NEW, facecolor=PANEL_BG, zorder=0)
            ax_graph.add_patch(rect)
            ax_graph.text(0, 0.1, "Energy decreases ✓", color=EDGE_NEW, ha='center', fontsize=12, weight='bold')
            ax_graph.text(0, -0.1, "Confidence: 0.82", color=EDGE_ACTIVE, ha='center', fontsize=12, weight='bold')
            
        elif frame < 42:
            title = "Graph repaired — no retraining needed"
            leak = max(0.05, 0.57 - ((frame - 34) / 8) * 0.52)
            energy = max(3.5, 7.7 - ((frame - 34) / 8) * 4.2)
            current_edges.append((3, 1))
            draw_edges(ax_graph, base_edges, EDGE_ACTIVE)
            draw_edges(ax_graph, [(3, 1)], EDGE_NEW, width=3)
            draw_nodes(ax_graph, node_colors)
            ax_graph.text(0, -1.3, "Added edge 4→2: thermal feedback coupling detected", 
                          color=EDGE_NEW, fontsize=11, ha='center', weight='bold')
            
        elif frame < 50:
            title = "Causal equilibrium restored"
            leak = np.random.uniform(0.02, 0.05)
            energy = 3.5 + np.random.uniform(-0.1, 0.1)
            current_edges.append((3, 1))
            draw_edges(ax_graph, current_edges, EDGE_ACTIVE)
            draw_nodes(ax_graph, node_colors)
            
        elif frame < 55:
            title = "Every change is traceable"
            leak = 0.03
            energy = 3.5
            current_edges.append((3, 1))
            draw_edges(ax_graph, current_edges, EDGE_ACTIVE)
            draw_nodes(ax_graph, node_colors)
            
        else:
            title = "Living causal graphs for any domain"
            leak = 0.03
            energy = 3.5
            ax_graph.text(0, 0.2, "pip install causalnerve", color=TEXT_COLOR, fontsize=20, ha='center', weight='bold')
            ax_graph.text(0, -0.2, "github.com/agency/CausalNerve", color=EDGE_ACTIVE, fontsize=16, ha='center')
        
        leakage_hist.append(leak)
        energy_hist.append(energy)
        
        ax_leakage.plot(leakage_hist, color=NODE_ALARM if leak > 0.15 else EDGE_NEW, linewidth=2)
        ax_leakage.set_xlim(0, 60)
        ax_leakage.set_ylim(0, 0.7)
        ax_leakage.tick_params(colors=TEXT_COLOR, labelsize=8)
        
        ax_energy.plot(energy_hist, color=EDGE_ACTIVE, linewidth=2)
        ax_energy.set_xlim(0, 60)
        ax_energy.set_ylim(2, 9)
        ax_energy.tick_params(colors=TEXT_COLOR, labelsize=8)
        
        if frame >= 35:
            ax_audit.text(0.05, 0.8, "Revision 1042", color=TEXT_COLOR, fontsize=9, weight='bold')
            ax_audit.text(0.05, 0.6, "Time: Cycle 42,912", color=TEXT_COLOR, fontsize=8)
            ax_audit.text(0.05, 0.4, "Action: ADD 4→2", color=EDGE_NEW, fontsize=9, weight='bold')
            ax_audit.text(0.05, 0.2, "Conf: 0.82 | Leak: -18%", color=TEXT_COLOR, fontsize=8)
            ax_audit.text(0.05, 0.0, "Reason: Dual-world validation passed", color=TEXT_COLOR, fontsize=8)
            
        title_text.set_text(title)
        
    return fig, ax_graph, ax_leakage, ax_energy, ax_audit, title_text, pos, base_edges, update, FRAMES, FPS

if __name__ == "__main__":
    create_flagship_gif()
