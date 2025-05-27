# DCT API Assistant

## Setup and Run

### 1. Clone the project
```bash
git clone https://github.com/Sureshallamsetti/openapi-assistant.git
```

### 2. Change directory
```bash
cd openapi-assistant
```

### 3. Install required Python packages
```bash
pip install -r requirements.txt
```

### 4. Configure your environment
Set the following environment variables:
- `BASE_URL`
- `HEADERS`
- `OPENAI_API_KEY`

You can export them in your terminal or store them in a `.env` file.

---

## Ollama Setup (Mac Users)

Before running `app.py`, you need to set up the Ollama server:

### 5. Install and Start Ollama
```bash
brew install ollama
ollama serve
```

This starts the Ollama server.

### 6. In a separate terminal, pull the model
```bash
ollama pull mistral-nemo:12b-instruct-2407-q4_0
```

---

## Continue Setup

### 7. Generate tools from the OpenAPI spec
```bash
python tools_generator.py api.yaml
```

This will create:
- `tools/generated_tools.py`
- `tools/functions_schema.json`

### 8. Generate tool embeddings
```bash
python vector_tools.py
```

This will generate:
- `vector/tool_embeddings.npy`
- `vector/tools.hash`
- `vector/tools.index`

### 9. Run the app
```bash
python app.py
```

### 10. Open the chat GUI in your browser
```
http://localhost:8300/
```

---

Feel free to contribute or report issues!
