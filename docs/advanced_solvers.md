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

## Optimization Performance Tuning

EMHASS includes several features to improve optimization performance, especially for complex configurations with many deferrable loads and battery systems.

### Object Caching & Warm Start

EMHASS automatically caches the optimization problem structure between runs. When the system configuration hasn't changed, subsequent optimizations reuse the cached problem structure and warm-start from the previous solution. This can significantly speed up repeated optimizations (e.g., during MPC operation).

The cache is automatically invalidated when:
- Number of deferrable loads changes
- Battery configuration changes
- Thermal load configuration changes
- Prediction horizon changes

No configuration is required—caching works automatically.

### MIP Gap Tolerance

For Mixed-Integer Programming problems (which occur when using semi-continuous loads, single-constant loads, or startup penalties), the solver can spend significant time proving a solution is exactly optimal. The `lp_solver_mip_rel_gap` parameter allows the solver to stop earlier when a "good enough" solution is found.

**Configuration:**

```yaml
# In config.yaml or config.json
lp_solver_mip_rel_gap: 0.05  # Stop when within 5% of optimal
```

**Valid range:** 0 to 1 (representing 0% to 100% gap tolerance). Values outside this range are clamped automatically.

**Recommended values:**

| Value | Description | Use Case |
|-------|-------------|----------|
| 0 | Exact optimal (default) | When precision is critical |
| 0.05 | Within 5% of optimal | **Recommended** for most users - ~2x speedup |
| 0.10 | Within 10% of optimal | Fast solving, good for testing |
| 0.20 | Within 20% of optimal | Very fast, adequate for simple decisions |

**Benchmarks show:**
- 5% gap: ~1.75x speedup
- 10% gap: ~1.86x speedup
- 20% gap: ~2.89x speedup

For home energy optimization, a 5% gap is typically imperceptible in practice—the difference between "optimal" and "within 5% of optimal" is usually smaller than forecast uncertainty.

**Note:** This parameter only affects problems with binary variables. Pure linear problems (continuous loads only, no battery) are unaffected.