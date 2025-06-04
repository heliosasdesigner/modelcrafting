# GPT-2-Style Model Crafting Lab

A pragmatic, experiment-driven project for building, training, and serving a GPT-2-style language model using modern ML tooling (PyTorch Lightning, MLflow, Streamlit, FastAPI, and more). This project is inspired by Karpathy's nanoGPT and is designed for rapid prototyping, robust experiment tracking, and easy production uplift.

---

## 🚀 Quickstart

### 0. Prep a Clean Workspace

1. **Create a fresh environment:**
   ```bash
   conda create -n gpt2-lab python=3.11
   conda activate gpt2-lab
   ```

2. **Install core libraries (adjust CUDA version as needed):**
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
   pip install pytorch-lightning streamlit mlflow wandb pydantic[extra] polars \
       postgresql sqlalchemy psycopg2-binary rich tqdm
   ```

3. **Clone Karpathy's reference repo for prototyping:**
   ```bash
   git clone https://github.com/karpathy/nanoGPT.git
   cd nanoGPT
   ```

---

## 1. Re-implement the Tutorial Model with Lightning

- **Tokenizer:**  
  Reuse `encode.py` from nanoGPT. Wrap in a Pydantic class for typed config (vocab size, etc).

- **Model:**  
  Copy `model.py` blocks (MultiHeadAttention, Block, etc.) and inherit from `torch.nn.Module`.  
  Wrap forward pass in a PyTorch-Lightning `LightningModule` (add `training_step`, `configure_optimizers`).

- **Config:**  
  Create `config.py` with a `GPTConfig` Pydantic model (`n_layer`, `n_head`, `n_embd`, `dropout`, `vocab_size`, `ctx_len`).

- **Dataset:**  
  Start with `tiny_shakespeare.txt` (~10MB) to confirm training loop.

- **Trainer Script:**  
  In `train.py`, parse a Pydantic config, instantiate LightningModule/DataModule, and call `pl.Trainer`.  
  Add `MLflowLogger()` or `WandbLogger()`.

- **Experiment Metadata:**  
  Open an MLflow run inside the Streamlit callback that launches training (`mlflow.start_run()`), so every widget tweak is logged.

---

## 2. Streamlit Front-End (Phase 1)

- **Create `app.py`:**
  - Sidebar widgets: dataset path, context length, batch size, learning rate, max iters, etc.
  - "Train" button: builds `GPTConfig`, spawns Lightning Trainer (GPU, mixed precision), pipes logs to MLflow/W&B, appends run metadata to PostgreSQL via SQLAlchemy.
  - Live progress: use `st.progress` and `st.line_chart` (stream metrics via Lightning's CSVLogger or MLflow).

---

## 3. Production-Grade Uplift (Phase 2 Roadmap)

- **Package Refactor:**  
  Move model/tokenizer into `ml_app/`. Expose `predict(text, max_new_tokens)`.

- **Serving:**  
  Spin up FastAPI in `api/`. POST `/generate` receives JSON, calls `predict`. Mount OpenAPI docs.

- **Background Jobs:**  
  Use Celery (with Redis) for scheduled retraining and batch scoring. FastAPI enqueues tasks.

- **Model Registry:**  
  Point MLflow to PostgreSQL storage; promote best checkpoints to Staging/Production.

- **Docker & Orchestration:**  
  Write `docker-compose.yml` for dev; for prod, bake images and deploy to k8s/ECS.

- **Observability:**  
  Instrument Prometheus metrics in FastAPI middleware; dashboard in Grafana.

- **Operator UI:**  
  If Streamlit suffices, keep it. Otherwise, iterate toward a React + Tailwind dashboard.

---

## 4. Validation & Sanity Checks

- Overfit a single batch (loss → 0).
- Greedy sample after every N steps.
- Compare perplexity on held-out split vs. nanoGPT.
- Profiling: `torch.cuda.memory_summary()`, Lightning's Profiler.

---

## 5. Nice-to-Haves

- **DAG-style orchestration:** Prefect 2.
- **Data versioning:** DVC or LakeFS.
- **Real-time log streaming:** Socket.IO from FastAPI.



---

## 📚 References

- [nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [MLflow](https://mlflow.org/)
- [Streamlit](https://streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Celery](https://docs.celeryq.dev/)
- [Prefect](https://www.prefect.io/)
- [DVC](https://dvc.org/)
- [LakeFS](https://lakefs.io/)

---

## 📝 Notes

- For learning-rate schedules, batch-size scaling, and evaluation frequency, see the "eight timeless tips" video (YouTube ID: iCwvGys_iM4).
- Bake best practices into Lightning callbacks (e.g., `LearningRateMonitor`, `ModelCheckpoint`). 