
# RAPID Pipeline - Parallelization Guide

**Auto-Adaptive Parallel Processing for Multi-Hardware Environments**

---

## 🎯 Overview

The RAPID pipeline now automatically detects your hardware and optimizes parallelization settings for maximum performance across different systems:

- **Laptop** (16 GB RAM, 4-8 cores): Conservative, reliable execution
- **Workstation** (32-64 GB RAM, 8-16 cores): Balanced performance
- **Server** (128 GB RAM, 32 cores): Maximum parallelization

---

## 🖥️ Hardware Profiles

### Laptop Profile (<20 GB RAM)
```
Specs: 16 GB RAM, 4-8 cores
Strategy: Conservative to prevent memory exhaustion

Settings:
├─ Parallel Jobs: 2-4 (RAM limited)
├─ RAM per job: ~2.5 GB
├─ Hyperparameter iterations: 10
├─ Pre-dispatch: 2*n_jobs (staged execution)
└─ Total CV fits per model: 50 (10 configs × 5 folds)

Expected Runtime:
├─ Feature selection: 5-10 minutes
├─ Model training (8 models): 15-30 minutes
├─ Hyperparameter tuning: 30-60 minutes per model
└─ Total pipeline: 2-3 hours
```

### Workstation Profile (20-64 GB RAM)
```
Specs: 32-64 GB RAM, 8-16 cores
Strategy: Balanced for optimal throughput

Settings:
├─ Parallel Jobs: 6-12 (balanced)
├─ RAM per job: ~2.0 GB
├─ Hyperparameter iterations: 20
├─ Pre-dispatch: 2*n_jobs
└─ Total CV fits per model: 100 (20 configs × 5 folds)

Expected Runtime:
├─ Feature selection: 3-5 minutes
├─ Model training (8 models): 10-15 minutes
├─ Hyperparameter tuning: 15-30 minutes per model
└─ Total pipeline: 1-2 hours
```

### Server Profile (64+ GB RAM)
```
Specs: 128 GB RAM, 32 logical cores
Strategy: Aggressive parallelization for maximum speed

Settings:
├─ Parallel Jobs: 28 (28 out of 32 cores, reserve 4 for OS)
├─ RAM per job: ~1.5 GB
├─ Hyperparameter iterations: 100
├─ Pre-dispatch: 'all' (dispatch everything at once)
└─ Total CV fits per model: 500 (100 configs × 5 folds)

Expected Runtime:
├─ Feature selection: 1-2 minutes
├─ Model training (8 models): 3-5 minutes
├─ Hyperparameter tuning: 8-15 minutes per model
└─ Total pipeline: 30-60 minutes

Performance Gains:
├─ vs Laptop: 3-4x faster
├─ vs Workstation: 2x faster
└─ Parallel efficiency: ~85-90% (28 cores utilized)
```

---

## ⚙️ How It Works

### Automatic Detection (in `config.py`)

```python
import psutil
from multiprocessing import cpu_count

def detect_hardware_capabilities():
    total_cores = cpu_count()
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # Classify hardware profile
    if total_ram_gb < 20:
        profile = "LAPTOP"
    elif total_ram_gb < 64:
        profile = "WORKSTATION"
    else:
        profile = "SERVER"
    
    # Set optimal parallel jobs
    # Account for: available RAM ÷ estimated RAM per job
    # Cap at usable CPU cores
```

### Configuration Constants

All parallelization settings in `config.py`:

```python
# Auto-detected settings (example for 32-core server with 128 GB RAM)
N_JOBS = 28                        # Parallel jobs
HYPERPARAM_SEARCH_ITER = 100       # Search iterations
PARALLEL_BACKEND = 'loky'          # Backend (robust for ML)
PARALLEL_PRE_DISPATCH = 'all'      # Aggressive dispatch
PARALLEL_VERBOSE = 0               # Silent (change to 1 for progress)
```

---

## 🚀 Where Parallelization Is Applied

### 1. Cross-Validation (Biggest Impact)

**What:** Each CV fold trains on different data split

**Parallelization:**
```python
cross_val_score(
    model, X, y, 
    cv=5,                    # 5 folds
    n_jobs=config.N_JOBS,    # 28 parallel jobs on server
    scoring='r2'
)
```

**Impact:**
- **Sequential:** 5 folds × 30 sec = 2.5 minutes
- **Parallel (28 jobs):** All 5 folds complete in ~30 seconds
- **Speedup:** 5x faster

### 2. Hyperparameter Search

**What:** Test different parameter combinations

**Parallelization:**
```python
RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=config.HYPERPARAM_SEARCH_ITER,  # 100 on server
    cv=5,                                   # 5-fold CV
    n_jobs=config.N_JOBS,                   # 28 parallel
    scoring='r2'
)
```

**Impact (Server with 100 iterations):**
- Total fits: 100 configs × 5 folds = **500 fits**
- Sequential time: 500 × 5 sec = **42 minutes**
- Parallel (28 jobs): 500 ÷ 28 = 18 batches × 5 sec = **~2 minutes**
- **Speedup:** 20x faster!

### 3. Feature Importance (Multiple Models)

**What:** Train 4 models (RF, XGB, GB, CatBoost) to rank features

**Parallelization:**
```python
# Each model's internal parallelization
RandomForestRegressor(n_jobs=config.N_JOBS)
XGBRegressor(n_jobs=config.N_JOBS)
GradientBoostingRegressor()  # Sequential by design
CatBoostRegressor(thread_count=config.N_JOBS)
```

**Impact:**
- Random Forest: 100 trees built in parallel
- XGBoost: Parallel tree construction
- Speedup: 5-10x faster per model

### 4. Imputation (Advanced Methods)

**What:** KNN and Iterative imputation

**Parallelization:**
```python
KNNImputer(n_neighbors=5)
IterativeImputer(max_iter=10)  # Internal parallelization in estimator
```

**Impact:**
- Moderate speedup (2-3x)
- Memory-bound more than CPU-bound

---

## 📊 Expected Performance Gains

### Server (128 GB, 32 cores) vs Laptop (16 GB, 4 cores)

| Operation | Laptop Time | Server Time | Speedup |
|-----------|-------------|-------------|---------|
| Feature Selection (4 models) | 10 min | 2 min | **5x** |
| Single Model CV (5-fold) | 2 min | 20 sec | **6x** |
| Hyperparameter Search (1 model) | 60 min | 3 min | **20x** |
| Full Pipeline (8 models) | 3 hours | 45 min | **4x** |

### Why Not Linear Speedup (32x)?

**Overhead factors:**
1. **Memory bandwidth:** 28 jobs competing for RAM access
2. **I/O bottleneck:** Data loading/saving is sequential
3. **Python GIL:** Some operations still single-threaded
4. **Coordination cost:** Managing 28 parallel processes
5. **Diminishing returns:** Beyond 16-20 cores, efficiency drops

**Realistic efficiency:** 80-90% for well-parallelized operations

---

## 🔧 Manual Override Options

### Force Specific Number of Jobs

Edit `config.py`:
```python
# Override auto-detection (use with caution)
N_JOBS = 16  # Force 16 parallel jobs regardless of hardware
```

### Disable Parallelization (Debugging)

```python
N_JOBS = 1  # No parallelization
PARALLEL_VERBOSE = 2  # Show detailed debug info
```

### Increase Hyperparameter Search

```python
# For exhaustive search on server
HYPERPARAM_SEARCH_ITER = 200  # 200 configs × 5 folds = 1,000 fits
# With 28 jobs: ~5 minutes per model
```

---

## ⚠️ Troubleshooting

### Issue: Out of Memory Errors

**Symptoms:** Process killed, "MemoryError", system freezes

**Solutions:**
1. Reduce `N_JOBS` manually in `config.py`
2. Increase `PARALLEL_PRE_DISPATCH` to `'2*n_jobs'` (stages execution)
3. Reduce dataset size or feature count

```python
# Conservative settings
N_JOBS = max(1, config.N_JOBS // 2)  # Use half the cores
PARALLEL_PRE_DISPATCH = '2*n_jobs'   # Stage execution
```

### Issue: Slow Performance Despite Many Cores

**Possible causes:**
1. **I/O bound:** Data loading is the bottleneck
2. **Small dataset:** Parallelization overhead exceeds benefit
3. **Memory bandwidth:** All cores waiting for RAM

**Check:**
```python
# Monitor during execution
import psutil
print(f"CPU usage: {psutil.cpu_percent(interval=1)}%")
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

**Solutions:**
- If CPU < 70%: I/O or memory bound, reduce jobs
- If Memory > 90%: Out of RAM, reduce jobs
- If both high: Working efficiently!

### Issue: "Too Many Open Files"

**Cause:** Parallel jobs opening many file handles

**Solution (Linux/Mac):**
```bash
ulimit -n 4096  # Increase file descriptor limit
```

**Solution (Windows):** Usually not an issue

---

## 📈 Monitoring Parallel Execution

### Enable Progress Tracking

In `config.py`:
```python
PARALLEL_VERBOSE = 1  # Show progress bars for parallel jobs
```

Output example:
```
[Parallel(n_jobs=28)]: Using backend LokyBackend with 28 concurrent workers.
[Parallel(n_jobs=28)]: Done  42 tasks      | elapsed:    3.2s
[Parallel(n_jobs=28)]: Done 192 tasks      | elapsed:   14.1s
[Parallel(n_jobs=28)]: Done 442 tasks      | elapsed:   32.7s
[Parallel(n_jobs=28)]: Done 500 out of 500 | elapsed:   37.5s finished
```

### Real-Time Resource Monitoring

```python
import psutil
import time

def monitor_resources(duration_sec=60):
    """Monitor CPU and RAM usage during pipeline execution."""
    for i in range(duration_sec):
        cpu_pct = psutil.cpu_percent(interval=1, percpu=False)
        ram_pct = psutil.virtual_memory().percent
        print(f"[{i:3d}s] CPU: {cpu_pct:5.1f}%  |  RAM: {ram_pct:5.1f}%")
        time.sleep(1)

# Run in separate terminal while pipeline executes
monitor_resources(duration_sec=300)  # Monitor for 5 minutes
```

---

## 🎓 Best Practices

### For Laptops (16 GB RAM)
✅ Use default auto-detected settings  
✅ Close other applications during training  
✅ Consider reducing `HYPERPARAM_SEARCH_ITER` to 5 for quick iterations  
❌ Don't force high `N_JOBS` values  

### For Workstations (32-64 GB RAM)
✅ Can increase `HYPERPARAM_SEARCH_ITER` to 30-50  
✅ Safe to run other light applications  
✅ Monitor first run to tune settings  
⚠️ Watch for memory usage spikes  

### For Servers (128+ GB RAM)
✅ Maximum parallelization enabled by default  
✅ Can run multiple pipelines simultaneously  
✅ Consider `HYPERPARAM_SEARCH_ITER = 200` for production  
✅ Enable verbose logging for production runs  
⚠️ Still reserve 4+ cores for OS/services  

---

## 📚 Technical Details

### Backend Comparison

| Backend | Pros | Cons | Use Case |
|---------|------|------|----------|
| **loky** (default) | Robust, handles crashes well | Slight overhead | Production, default |
| **multiprocessing** | Fast, efficient | Can hang on errors | Stable code only |
| **threading** | Low overhead | GIL limits parallelism | I/O-bound tasks |

### Memory Estimation Formula

```
Total RAM needed = Base + (N_JOBS × RAM_per_job)

Where:
- Base = Dataset size × 2-3 (for copies during processing)
- RAM_per_job = Model size + CV fold data + overhead
- Model size ≈ 100 MB - 2 GB (depends on algorithm)
- CV fold data = dataset_size / CV_FOLDS
- Overhead ≈ 500 MB per job (Python, libraries, etc.)

Example (Server with 1 GB dataset, 28 jobs, RF model):
- Base: 1 GB × 3 = 3 GB
- RAM_per_job: 200 MB (model) + 200 MB (fold) + 500 MB = 0.9 GB
- Total: 3 + (28 × 0.9) = 28 GB
- Safe with 128 GB total (22% utilization)
```

---

## 🚀 Future Enhancements

### Potential Improvements

1. **GPU Acceleration** (for supported models)
   ```python
   XGBRegressor(tree_method='gpu_hist')  # Use CUDA GPU
   ```

2. **Distributed Computing** (across multiple servers)
   ```python
   from dask.distributed import Client
   client = Client('scheduler-address:8786')
   ```

3. **Adaptive Job Sizing** (dynamic based on task)
   ```python
   # Use more jobs for simple models, fewer for complex
   n_jobs = get_optimal_jobs(model_complexity, ram_available)
   ```

4. **Cloud Integration** (AWS/Azure parallel execution)

---

## ✅ Validation Checklist

Before running on new hardware:

- [ ] Check auto-detected settings: `config.print_config()`
- [ ] Verify `N_JOBS` is reasonable (not > total cores)
- [ ] Monitor first run for memory usage
- [ ] Compare actual vs expected runtime
- [ ] Check logs for parallel execution confirmation
- [ ] Validate results match sequential execution (spot check)

---

**Last Updated:** December 1, 2025  
**Version:** 2.0  
**Status:** Production Ready - Auto-Adaptive Parallelization ✅
