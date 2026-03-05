from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import query, documents, auth, billing, admin
from src.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="ManualAI",
        description="Conversational search over 400K+ digital manuals",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router, prefix="/v1/auth", tags=["auth"])
    app.include_router(query.router, prefix="/v1/query", tags=["query"])
    app.include_router(documents.router, prefix="/v1/documents", tags=["documents"])
    app.include_router(billing.router, prefix="/v1/billing", tags=["billing"])
    app.include_router(admin.router, prefix="/v1/admin", tags=["admin"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
