import logging
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

# Import Routes and Middleware
from app.api import routes_predict, routes_auth
from app.middleware.logging_middleware import LoggingMiddleware
from app.core.exceptions import register_exception_handlers
from app.core.config import settings

# 1. Configure Global Logging
# This ensures that your 'logging.info' calls inside the middleware actually print to the console/logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# 2. Initialize FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="An intelligent API to predict wine quality using tuned ML models (XGBoost, LightGBM, CatBoost).",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI endpoint
    redoc_url="/redoc" # ReDoc endpoint
)

# 3. Register Middleware
# Logs every request, status code, and execution time
app.add_middleware(LoggingMiddleware)

# 4. Register Routes
# We separate Auth and Prediction logic for cleanliness
app.include_router(routes_auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(routes_predict.router, tags=["Prediction"])

# 5. Add Monitoring (Prometheus)
# Exposes metrics at /metrics for tools like Grafana
Instrumentator().instrument(app).expose(app)

# 6. Register Custom Exception Handlers
# Ensures pretty JSON errors instead of server crashes
register_exception_handlers(app)

# 7. Startup Event
@app.on_event("startup")
async def startup_event():
    """
    Runs when the API server starts.
    Good place to check connections (Redis, DB) or print status.
    """
    logger.info("🍷 WineMeUp API is starting up...")
    logger.info(f"Environment: {settings.ENV}")
    logger.info("Ready to accept connections.")