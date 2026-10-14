import pathlib
import jinja2
from typing import Dict

def render_slurm_template(template_path: str, output_path: str, context: Dict):
    """Renders a Jinja2 template (.slurm.j2) into an active Slurm batch script (.slurm).

    Args:
        template_path (str): Path of the template.
        output_path (str): Destination filename for the rendered slurm script.
        context (Dict): Dictionary of template variables.

    Returns:
        pathlib.Path: Path to the written Slurm batch script.
    """
    template_file = pathlib.Path(template_path)
    output_file = pathlib.Path(output_path)
    
    if not template_file.exists():
        raise FileNotFoundError(f"Slurm template not found: {template_file}")
        
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(template_file, "r", encoding="utf-8") as f:
        template = jinja2.Template(f.read())

    rendered = template.render(**context)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(rendered)

    print(f"Rendered Slurm script: {output_file.name} from {template_file.name}")
    return output_file

def ensure_slurm_log_dirs(scratch_root: str, subdirs=None):
    """Creates the standard log directories inside the root folder to prevent slurm errors."""
    scratch_logs = pathlib.Path(scratch_root) / "logs"
    scratch_logs.mkdir(parents=True, exist_ok=True)
    if subdirs:
        for subdir in subdirs:
            (pathlib.Path(scratch_root) / subdir / "logs").mkdir(parents=True, exist_ok=True)
