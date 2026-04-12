"""
FastAPI server for Crisis Response Environment (OpenEnv compliant)

"""

from openenv.core.env_server.http_server import create_app
import uvicorn

# Import your models and environment
from models import MyAction, MyObservation
from .my_env_environment import MyEnvironment

# Create FastAPI app

app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name= "crisis_response_env",
    max_concurrent_envs=5,
)

def main():
     """
    Entry point for running server locally or via Docker

    """
     uvicorn.run("server.app:app",
                  host="0.0.0.0",
                    port=7860
                    )

if __name__ == "__main__":
    main()
     
