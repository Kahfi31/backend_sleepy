import logging
from fastapi import FastAPI, APIRouter
from . import models, schemas, database
from dotenv import load_dotenv
import redis.asyncio as aioredis
from backend.oauth2.authroutes import auth_router
from .predictroutes import predict_router
from .userroutes import user_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database initialization
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()
redis = None
load_dotenv()
router = APIRouter()
otp_storage = {}

app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(user_router)

@app.on_event("startup")
async def startup_event():
    global redis
    redis = await aioredis.from_url("redis://localhost", decode_responses=True)  # Membuat pool dengan from_url

@app.post("/store-info")
async def store_user_info(user_info: schemas.UserInfo):
    user_data = user_info.dict()
    await redis.set(user_info.gender, str(user_data)) 
    return {"message": "Data stored successfully in Redis"}
