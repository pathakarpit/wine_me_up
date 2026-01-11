# 🍷 WineMeUp: High-Performance Multi-Model Inference Ecosystem

**Live Production Site:** [project.patech.in/datascience/winemeup/](http://project.patech.in/datascience/winemeup/)



## 🎯 The Vision
Most data science projects never leave a Jupyter Notebook. **WineMeUp** is a demonstration of how to bridge the gap between a "working model" and a "scalable production system." It is a containerized, microservices-based architecture designed for high-availability, low-latency wine quality predictions.

This project doesn't just predict a score; it implements a robust **Inference Engine** capable of handling model routing, multi-layer caching, and real-time observability.

---

## 🏗️ Architectural Excellence

The system is engineered as a decoupled stack to ensure independent scalability of the UI and the Inference API.



### **Core Components:**
* **Inference API (FastAPI):** An asynchronous backend designed for high-throughput. It features strict Pydantic schema validation and dynamic model loading.
* **Intelligent Caching (Redis):** Implemented a deterministic MD5-hashing mechanism for input data. Repeat queries return in **<2ms**, bypassing the ML model entirely to save compute costs.
* **Dynamic Model Routing:** The system supports hot-swapping between 7 different pre-trained architectures including **XGBoost, LightGBM, CatBoost, and Random Forest**.
* **Observability Stack (Prometheus):** Real-time monitoring of API health, inference latency, and Redis cache hit ratios.
* **Reverse Proxy (Nginx):** Manages SSL termination and WebSocket headers, ensuring secure and stable connections to the Streamlit frontend.

---

## 🧠 Model Deep Dive: The "Secret Weapon"
While the system supports multiple models, the **LightGBM Regressor** serves as the primary engine. 

### **Technical Specifics:**
* **Optimization:** Hyperparameters were tuned using **Optuna** with a focus on minimizing Root Mean Squared Error (RMSE).
* **Data Alignment:** Implemented a custom mapping layer to resolve the "feature-name mismatch" common in production pipelines where API field names (underscored) differ from training headers (spaced).
* **Post-Processing:** The engine handles raw output unwrapping (addressing NumPy array nesting issues specific to CatBoost) to ensure clean integer quality scores.

---

## 🛠️ The Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.12 |
| **ML Frameworks** | Scikit-Learn, XGBoost, LightGBM, CatBoost |
| **API Framework** | FastAPI (Uvicorn) |
| **Frontend** | Streamlit |
| **Caching/DB** | Redis 7.0 |
| **DevOps/Ops** | Docker, Docker Compose, Nginx |
| **Monitoring** | Prometheus, Redis Exporter |

---

## 🚀 Deployment & Reproducibility

The entire stack is containerized. To replicate this production environment:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/WineMeUp.git](https://github.com/your-username/WineMeUp.git)
    cd WineMeUp
    ```

2.  **Environment Configuration:**
    Create a `.env` file with your `API_KEY` and `MODEL_DIR`.

3.  **One-Click Launch:**
    ```bash
    docker-compose up --build -d
    ```

4.  **Verify Services:**
    * **Frontend:** `http://localhost:8501`
    * **Inference API Docs:** `http://localhost:8000/docs`
    * **Prometheus Metrics:** `http://localhost:9090`

---

## 📈 Engineering Highlights

* **Lazy-Loading Pattern:** Models are only loaded into RAM when first called, optimizing memory usage for a multi-model environment.
* **Zero-Downtime Design:** The use of Docker Compose and Nginx allows for service updates without interrupting user experience.
* **Production Guardrails:** Implemented robust error handling for API authentication failures and inference-time array dimension mismatches.

---

## 👤 About the Developer
I am a Data Scientist & Machine Learning Engineer focused on the intersection of **Mathematical Modeling** and **Scalable Software Architecture**. I build tools that don't just work—they perform.

**Connect with me:** [LinkedIn](https://linkedin.com/in/yourprofile) | [Portfolio](https://patech.in)