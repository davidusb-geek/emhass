# Using Commercial Solvers (CPLEX / Gurobi)

By default, EMHASS uses the high-performance open-source solver **HiGHS**. It is bundled with the default Docker image and requires no configuration.

If you possess a license for a commercial solver like **Gurobi** or **CPLEX**, you can enable them by building a custom Docker image.

```{note} 

This functionality is intended for advanced users. This will assume using the Docker Standalone installation method. See [Installation methods](installation_methods) section for more details.
```

## 1. Create a Custom Dockerfile

Create a file named `Dockerfile.custom` that extends the official EMHASS image.

**For Gurobi:**

```dockerfile
FROM davidusb-geek/emhass:latest

# Install the Gurobi Python interface
RUN uv pip install gurobipy

# Set the solver environment variable
ENV LP_SOLVER=GUROBI

```

**For CPLEX:**

```dockerfile
FROM davidusb-geek/emhass:latest

# Install the CPLEX Python interface
RUN uv pip install cplex

# Set the solver environment variable
ENV LP_SOLVER=CPLEX

```

## 2. Build the Image

Build your custom image locally:

```bash
docker build -t emhass-custom -f Dockerfile.custom .

```

## 3. Run with License Mounting

When running the container, you must mount your license file to the location expected by the solver.

**Example for Gurobi:**
Assuming your license file is at `/home/user/gurobi.lic`:

```bash
docker run -d \
  --name emhass \
  -p 5000:5000 \
  -v /home/user/config_emhass.json:/app/config_emhass.json \
  -v /home/user/gurobi.lic:/opt/gurobi/gurobi.lic \  <-- Mount License
  -e GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \       <-- Tell Gurobi where it is
  emhass-custom

```

**Note:** You do not need to change any configuration in `config.json` or the web UI. The `LP_SOLVER` environment variable handles the switch automatically.