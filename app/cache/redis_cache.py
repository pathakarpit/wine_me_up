import json
import redis
from app.core.config import settings

redis_client = redis.Redis.from_url(settings.REDIS_URL)

def get_cached_prediction(key: str):
    value = redis_client.get(key)
    if value:
        return json.loads(value)
    return None

def set_cached_prediction(key: str, value: dict, expire: int = 3600):
    redis_client.set(key, json.dumps(value), ex=expire)