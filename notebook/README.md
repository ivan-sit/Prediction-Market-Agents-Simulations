# Notebooks

Interactive Jupyter notebooks for testing and visualization.

## Available Notebooks

### `lmsr_pricing_test.ipynb`
Tests the LMSR (Logarithmic Market Scoring Rule) pricing mechanism.

**What it does:**
- Creates an LMSR market with liquidity parameter b=100
- Simulates 30 buy/sell trades
- Tracks and visualizes price evolution
- Saves results to `pricing_test/`

**How to run:**

```bash
# Activate virtual environment
source venv/bin/activate

# Start Jupyter (in browser)
jupyter notebook notebook/lmsr_pricing_test.ipynb --NotebookApp.token='' --NotebookApp.password=''

# Or use JupyterLab
jupyter lab notebook/lmsr_pricing_test.ipynb --NotebookApp.token=''
```

**Or open in VS Code:**
1. Install Jupyter extension
2. Open `notebook/lmsr_pricing_test.ipynb`
3. Click "Run All"

## Output

Results saved to:
- `pricing_test/price_data.csv` - Raw timestep data
- `pricing_test/price_evolution.png` - Price visualization

