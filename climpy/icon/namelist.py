import re
import pathlib
import stat
from typing import Dict, Optional

def patch_icon_runscript(template_path: str, output_path: str, replacements: Dict[str, str], append_lines: str = ""):
    """
    Reads an ICON runscript template, applies a dictionary of replacements using regex, 
    appends any additional configuration lines, and writes out the executable runscript.
    
    Args:
        template_path (str): Path to the base template runscript.
        output_path (str): Path where the modified runscript will be written.
        replacements (Dict[str, str]): Dictionary mapping regex patterns to replacement strings.
        append_lines (str): Additional bash commands or namelist patches to append before EOF.
    """
    template_file = pathlib.Path(template_path)
    output_file = pathlib.Path(output_path)
    
    if not template_file.exists():
        raise FileNotFoundError(f"Template run script not found at {template_file}")

    with open(template_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Apply all regex replacements
    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content)

    # If appending lines to the script, find where to append them.
    if append_lines:
        content += "\n# --- Injected by climpy.icon.namelist ---\n"
        content += append_lines + "\n"

    # Write out the customized runscript
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    # Make the destination script executable
    output_file.chmod(output_file.stat().st_mode | stat.S_IEXEC)
    print(f"Created executable ICON runscript: {output_file.resolve()}")
