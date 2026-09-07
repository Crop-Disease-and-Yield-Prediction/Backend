# Crop Backend

## Setup

1. Create and activate a Python virtual environment outside the repository's `venv` source folder:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` if the `/disease-cure` endpoint is needed.

4. Start the API from the `venv` source folder:

   ```powershell
   cd venv
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

The API will be available at `http://localhost:8000` and its documentation at `http://localhost:8000/docs`.