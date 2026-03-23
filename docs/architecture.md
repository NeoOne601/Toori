# System Architecture

The Toori system consists of several high‑level components:

- **Mobile app** – native iOS/Android client that captures images, runs on‑device inference, and communicates with the backend.
- **JEPA service** – cloud service that processes embeddings using the JEPA model.
- **FAISS service** – provides similarity search over embeddings.
- **API gateway** – FastAPI gateway that authenticates requests and routes them to the appropriate services.

These components work together to enable edge‑cloud hybrid image retrieval.
