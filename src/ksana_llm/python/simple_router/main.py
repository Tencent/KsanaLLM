# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

"""Main entry point for the KsanaLLM Router service.

This module initializes and starts the FastAPI application that serves as the
coordination service for distributed NCCL communication between prefill and
decode nodes.
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api import api_router
from api.endpoints import generate
from config import settings
from db import db

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("comm_coordinator")


def create_default_cluster():
    """Create the default cluster if it doesn't exist.

    This function checks if the default cluster (as specified in settings)
    exists, and if not, creates it.
    """
    # Get default cluster name from config
    cluster_name = settings.cluster_name

    # Check if default cluster already exists
    if not db.storage.get_cluster(cluster_name):
        try:
            # Register cluster
            db.register_cluster(cluster_name)
            logger.info(f"Created default cluster '{cluster_name}'")
        except ValueError as e:
            # If cluster already exists, this is normal
            logger.info(f"Default cluster already exists: {str(e)}")
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Error creating default cluster: {str(e)}")
    else:
        logger.info(
            f"Default cluster '{cluster_name}' already exists, no need to create"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler.

    This function is called when the application starts and shuts down.

    Args:
        app: The FastAPI application instance.
    """
    # Code to run on startup
    create_default_cluster()
    logger.info("Service started, default cluster initialized")
    yield
    # Code to run on shutdown
    logger.info("Service shutting down")


# Create FastAPI application
app = FastAPI(
    title="NCCL Distributed Coordination Service",
    description="Service for coordinating NCCL distributed training communication groups and nodes",
    version="1.0.0",
    lifespan=lifespan,
)

# Register API routers
app.include_router(api_router, prefix=settings.api_prefix)
app.include_router(generate.raw_router)


def main():
    """Start the NCCL Coordination Service.

    This function initializes and starts the Uvicorn server with the
    specified configuration.
    """
    logger.info("Starting NCCL Coordination Service...")

    # Start Uvicorn server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9080,
        workers=1,
        reload=True,
        log_level="debug",
        use_colors=True,
    )


if __name__ == "__main__":
    main()
