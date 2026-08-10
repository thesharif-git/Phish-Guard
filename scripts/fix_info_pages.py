import os
import re

info_dir = r"e:\Ai Phishing Detector\templates\info"

for filename in os.listdir(info_dir):
    if not filename.endswith(".html"):
        continue
    
    filepath = os.path.join(info_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract the <style>...</style> block
    style_match = re.search(r'<style>.*?</style>', content, re.DOTALL)
    style_block = style_match.group(0) if style_match else ""

    # Extract the main content (between header and footer)
    body_match = re.search(r'</header>(.*?)<footer', content, re.DOTALL)
    if not body_match:
        print(f"Failed to match body in {filename}")
        continue
    
    body_content = body_match.group(1).strip()
    
    title_match = re.search(r'<title>(.*?)</title>', content)
    title = title_match.group(1) if title_match else "PhishGuard"

    new_content = f"""{{% extends "public_base.html" %}}
{{% block title %}}{title}{{% endblock %}}

{{% block extra_head %}}
{style_block}
{{% endblock %}}

{{% block content %}}
<div class="bg-white">
{body_content}
</div>
{{% endblock %}}
"""
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Fixed {filename}")
