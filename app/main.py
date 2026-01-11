from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.api import routes_predict, routes_auth
from app.middleware.logging_middleware import LoggingMiddleware
from app.core.exceptions import register_exception_handlers

app = FastAPI(
    title="WineMeUp API",
    description="An API to predict wine quality using various ML models.",
    version="1.0.0"
)
app.add_middleware(LoggingMiddleware)

app.include_router(routes_auth.router, tags=["Authentication"])
app.include_router(routes_predict.router, tags=["Prediction"])

Instrumentator().instrument(app).expose(app)

register_exception_handlers(app)