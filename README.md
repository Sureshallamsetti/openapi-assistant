# DCT API Assistant

## Setup and Run

1. Clone the project:
   ```bash
   git clone <your-repo-url>
   ```

2. Change directory:
   ```bash
   cd openai_assistant
   ```

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your environment:
   - Add the `base_url`, `headers`, and OpenAPI API key as needed.

5. Generate tools from the OpenAPI spec:
   ```bash
   python tools_generator.py api.yaml
   ```
   This will create `generated_tools.py` and `functions_schema.json` inside the `tools` directory.

6. Generate tool embeddings:
   ```bash
   python vector_tools.py
   ```
   This will generate `tool_embeddings.npy`, `tools.hash`, and `tools.index`.

7. Run the app:
   ```bash
   python app.py
   ```

8. Open the chat GUI in your browser:
   ```
   http://localhost:8300/
   ```

---

Feel free to contribute or report issues!
