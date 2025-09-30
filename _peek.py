from pathlib import Path
text = Path('src/Workers.py').read_text()
start = text.index('        if channel_count <= 1:')
print(text[start:start+800])
