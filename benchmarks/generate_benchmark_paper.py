import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from causalnerve.reporting.paper_generator import PaperGenerator

def run_paper_generation():
    print("==================================================")
    print(" AUTOMATED SCIENTIFIC PAPER GENERATOR ")
    print("==================================================")
    
    output_dir = "results/generated_paper"
    generator = PaperGenerator(output_dir)
    
    print("[*] Generating LaTeX Source (paper.tex)...")
    generator.generate_latex()
    
    print("[*] Generating Markdown Summary (paper.md)...")
    generator.generate_markdown()
    
    print("[*] Bundling arXiv Submission (arxiv_bundle.zip)...")
    generator.generate_arxiv_bundle()
    
    print(f"[SUCCESS] Publication-ready artifacts compiled in '{output_dir}/'")

if __name__ == "__main__":
    run_paper_generation()
