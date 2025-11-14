# Notebooks

Interactive Jupyter notebooks for testing and visualization.

## Available Notebooks

### `lmsr_pricing_test.ipynb`
Tests the **LMSR** (simple, always liquid) pricing mechanism.

**What it does:**
- Creates an LMSR market with liquidity parameter b=100
- Simulates 30 buy/sell trades
- ALL orders execute (automated market maker)
- Tracks and visualizes price evolution
- Saves results to `pricing_test/`

### `orderbook_pricing_test.ipynb` ⭐ NEW!
Tests the **Order Book** (realistic, like Kalshi) pricing mechanism.

**What it does:**
- Creates an Order Book market with price-time priority
- Simulates 20 limit/market orders
- SOME orders FAIL (no counterparty)
- Shows bid-ask spread and execution rate
- Visualizes order success/failure
- Saves results to `orderbook_test/`

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

