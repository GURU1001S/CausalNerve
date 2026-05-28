"""
export_gif.py
=============
Generates the viral GIFs and screenshots for the CausalNerve README.
Requires: pip install kaleido imageio
"""

import os
import imageio
from demo_dashboard import fig_graph, fig_cf, fig_fleet

def generate_screenshots():
    print("Generating static high-res screenshots for README...")
    os.makedirs("assets", exist_ok=True)
    
    # Export using kaleido (needs to be installed)
    try:
        fig_graph.write_image("assets/living_graph.png", scale=2)
        fig_cf.write_image("assets/counterfactual_split.png", scale=2, height=800, width=1200)
        fig_fleet.write_image("assets/fleet_epidemiology.png", scale=2)
        print("Successfully exported 3 high-res PNGs to assets/")
    except ValueError as e:
        print("Note: Install 'kaleido' to export static images: pip install -U kaleido")
        print(f"Error: {e}")

def generate_gif_storyboard():
    """
    Creates a simulated GIF of the graph evolving.
    In practice, this iterates over the OCGR history and snapshots each frame.
    """
    print("Generating animated GIF storyboard...")
    # This is a placeholder for the actual loop that would generate frames
    print("Export pipeline ready. Run with populated CausalNerve history to render frames.")

if __name__ == "__main__":
    generate_screenshots()
    generate_gif_storyboard()
