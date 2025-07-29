from jinja2 import Environment, FileSystemLoader
from itertools import product
import os

# Create the config file of experiment as the product of the following lists
backends = ["naive", "numba-aot", "numba-jit", "numpy"]
dimensions = [256, 512, 1024, 2048, 4096]
profilers = ["null", "line_profile", "memory_profile", "memray"]

# Default parameters
base_context = {
    "log_level": "INFO",
    "gen_random_matrices": True,
    "dtype": "float64",
    "generation_min": 0,
    "generation_max": 1000,
}

template_dir = '../experiments'  #Search for template in this folder
env = Environment(loader=FileSystemLoader(template_dir))  #Setup environment to search template in template_dir

# Set the output directory
output_dir = "../experiments"
os.makedirs(output_dir, exist_ok=True)
for backend in backends:
    os.makedirs(os.path.join(output_dir, backend), exist_ok=True)

# Load the template
template = env.get_template('config_template.yaml.j2')


for backend, dim, profiler in product(backends, dimensions, profilers):
    context = {
        **base_context,
        "dimensions": {
            "A": [dim, dim],
            "B": [dim, dim],
        },
        "backend": backend,
        "profiler": profiler,
    }
    rendered = template.render(context)
    filename = f"config_{backend}_{dim}x{dim}_{profiler}.yaml"
    filepath = os.path.join(output_dir,backend, filename)
    with open(filepath, "w") as f:
        f.write(rendered)
    print(f"Generated: {filepath}")
