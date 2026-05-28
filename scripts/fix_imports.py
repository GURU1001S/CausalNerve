import os

replacements = {
    "from causalnerve.adaptation": "from causalnerve.adaptationation",
    "import causalnerve.adaptation": "import causalnerve.adaptationation",
    "causalnerve.adaptation.": "causalnerve.adaptation.",
    
    "from causalnerve.reasoning ": "from causalnerve.reasoning ",
    "from causalnerve.reasoning.": "from causalnerve.reasoning.",
    "import causalnerve.reasoning ": "import causalnerve.reasoning ",
    "causalnerve.reasoning.": "causalnerve.reasoning.",
    
    "from causalnerve.core": "from causalnerve.core",
    "import causalnerve.core": "import causalnerve.core",
    "causalnerve.core.": "causalnerve.core.",

    "from causalnerve.runtime": "from causalnerve.runtime",
    "import causalnerve.runtime": "import causalnerve.runtime",
    "causalnerve.runtime.": "causalnerve.runtime.",

    "from causalnerve.runtime": "from causalnerve.runtime",
    "import causalnerve.runtime": "import causalnerve.runtime",
    "causalnerve.runtime.": "causalnerve.runtime.",

    "from causalnerve.config": "from causalnerve.config",
    "import causalnerve.config": "import causalnerve.config",
    "causalnerve.config.": "causalnerve.config.",

    "from causalnerve.plugins": "from causalnerve.plugins",
    "import causalnerve.plugins": "import causalnerve.plugins",
    "causalnerve.plugins.": "causalnerve.plugins.",

    "from causalnerve.reporting": "from causalnerve.reporting",
    "import causalnerve.reporting": "import causalnerve.reporting",
    "causalnerve.reporting.": "causalnerve.reporting.",

    "from causalnerve.visualization_stub": "from causalnerve.visualization_stub",
    "import causalnerve.visualization_stub": "import causalnerve.visualization_stub",
    "causalnerve.visualization_stub.": "causalnerve.visualization_stub.",

    "from causalnerve.memory.replay_engine": "from causalnerve.memory.replay_engine",
    "import causalnerve.memory.replay_engine": "import causalnerve.memory.replay_engine",
    "causalnerve.memory.replay_engine.": "causalnerve.memory.replay_engine.",
    
    "from causalnerve.memory.replay_engine": "from causalnerve.memory.replay_engine",
    "causalnerve.memory.replay_engine.": "causalnerve.memory.replay_engine.",
}

# Add replacements for relative imports inside causalnerve package
rel_replacements = {
    "from .adapt": "from .adaptation",
    "from .reason": "from .reasoning",
    "from .physics": "from .core",
    "from .live": "from .runtime",
    "from .streams": "from .runtime",
    "from .presets": "from .config",
    "from .domains": "from .plugins",
    "from .audit": "from .reporting",
    "from .viz": "from .visualization_stub",
    "from .replay_engine": "from .memory.replay_engine",
}

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    if "causalnerve\\" in filepath or "causalnerve/" in filepath:
        for old, new in rel_replacements.items():
            # ensure space or dot or import follows to avoid partial matches
            new_content = new_content.replace(old + " ", new + " ")
            new_content = new_content.replace(old + ".", new + ".")
            new_content = new_content.replace(old + "\n", new + "\n")

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

for root, dirs, files in os.walk('.'):
    if '.git' in root or '__pycache__' in root or '.pytest_cache' in root or 'outsider_venv' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            process_file(os.path.join(root, file))

print("Bulk rename complete.")
