import os

def check_corruption():
    template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
    corrupt_markers = ['âœ“', 'â€¦', 'Â·', 'Â©', 'Â']
    found_any = False

    for root, dirs, files in os.walk(template_dir):
        for f in files:
            if f.endswith(".html"):
                path = os.path.join(root, f)
                with open(path, 'rb') as file:
                    content = file.read()
                
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    text = content.decode('latin-1')
                
                for marker in corrupt_markers:
                    if marker in text:
                        print(f"STILL CORRUPT: {f} contains '{marker}'")
                        found_any = True
                        break
    
    if not found_any:
        print("All templates are clean. Zero encoding corruption detected.")

if __name__ == "__main__":
    check_corruption()
