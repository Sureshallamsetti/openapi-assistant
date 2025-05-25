import json
import keyword
import os
from dotenv import load_dotenv
import yaml
import re
from typing import Dict, List, Optional, Any

# Load environment variables
load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://localhost:8080/v3/")
DEFAULT_HEADERS = {
    "X-CLIENT-ID": "1",
    "Content-Type": "application/json"
}

TOOLS_JSON = "tools.json"
OUTPUT_PY = "generated_tools.py"

def sanitize_param_name(name: str) -> str:
    """Sanitize parameter names to be valid Python identifiers."""
    if keyword.iskeyword(name):
        return f"{name}_"
    return name.replace("-", "_")

def sanitize_func_name(name: str) -> str:
    """Sanitize function names to be valid Python identifiers."""
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if not sanitized.isidentifier() or keyword.iskeyword(sanitized):
        sanitized = f"func_{sanitized}"
    return sanitized

def extract_tools_from_openapi(openapi_path: str = "openapi.yaml") -> List[Dict]:
    """Extract API operations from OpenAPI specification."""
    with open(openapi_path, 'r') as f:
        spec = yaml.safe_load(f)

    tools = []
    paths = spec.get("paths", {})
    
    for path, methods in paths.items():
        for method, operation in methods.items():
            if method.lower() not in ("get", "post", "put", "delete", "patch"):
                continue

            operation_id = operation.get("operationId")
            if not operation_id:
                continue

            description = operation.get("description", "").strip() or f"{method.upper()} {path}"
            parameters = operation.get("parameters", [])
            request_body = operation.get("requestBody", {})
            props = {}
            required = []

            # Process parameters
            for param in parameters:
                name = param["name"]
                schema = param.get("schema", {})
                props[name] = {
                    "type": schema.get("type", "string"),
                    "description": param.get("description", ""),
                    "in": param.get("in", "query")
                }
                if param.get("required", False):
                    required.append(name)

            # Process request body
            if request_body:
                content = request_body.get("content", {})
                json_body = content.get("application/json", {}).get("schema", {})
                if "properties" in json_body:
                    for name, schema in json_body["properties"].items():
                        props[name] = {
                            "type": schema.get("type", "string"),
                            "description": schema.get("description", ""),
                            "in": "body"
                        }
                    required.extend(json_body.get("required", []))

            tools.append({
                "name": operation_id,
                "description": description,
                "method": method.lower(),
                "path": path,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(set(required)) if required else []
                }
            })

    with open(TOOLS_JSON, "w") as f:
        json.dump(tools, f, indent=2)

    print(f"✅ Extracted {len(tools)} tools from {openapi_path} → {TOOLS_JSON}")
    return tools

def generate_tool_function(tool: Dict) -> str:
    """Generate Python tool function from API operation definition."""
    name = sanitize_func_name(tool["name"])
    description = tool.get("description", "")
    method = tool.get("method", "get").lower()
    path = tool.get("path", "")
    param_definitions = tool["parameters"].get("properties", {})
    required = tool["parameters"].get("required", [])

    path_params = re.findall(r"\{(\w+)\}", path)
    required_params = []
    optional_params = []
    docstring_params = []

    # Add path parameters first (all required)
    for p in path_params:
        safe_name = sanitize_param_name(p)
        required_params.append(f"{safe_name}: str")
        docstring_params.append(f"    {safe_name} (str): Path parameter")

    # Process other parameters
    for param_name, schema in param_definitions.items():
        if param_name in path_params:
            continue
            
        safe_name = sanitize_param_name(param_name)
        param_type = {
            "string": "str",
            "integer": "int",
            "boolean": "bool",
            "number": "float"
        }.get(schema.get("type", "string"), "Any")
        
        param_desc = schema.get("description", "").strip() or param_name
        param_in = schema.get("in", "query")
        
        docstring_params.append(f"    {safe_name} ({param_type}): {param_desc} (in: {param_in})")
        
        if param_name in required:
            required_params.append(f"{safe_name}: {param_type}")
        else:
            optional_params.append(f"{safe_name}: Optional[{param_type}] = None")

    # Combine all parameters with required ones first
    func_params = required_params + optional_params
    
    # Prepare function signature
    if not func_params:
        param_str = "args: Optional[dict] = None"
    else:
        param_str = ", ".join(func_params)

    # Prepare URL path with parameters
    url_path = path.lstrip('/')
    for original in path_params:
        safe_name = sanitize_param_name(original)
        url_path = url_path.replace(f"{{{original}}}", f"{{{safe_name}}}")

    # Prepare request components
    query_params = []
    json_params = []
    path_params_set = set(path_params)

    for param_name, schema in param_definitions.items():
        safe_name = sanitize_param_name(param_name)
        if param_name in path_params_set:
            continue
            
        if schema.get("in") == "query" or method == "get":
            query_params.append(f'"{param_name}": {safe_name}')
        else:
            json_params.append(f'"{param_name}": {safe_name}')

    query_dict = "{" + ", ".join(query_params) + "}" if query_params else "None"
    json_dict = "{" + ", ".join(json_params) + "}" if json_params else "None"

    # Generate docstring
    docstring = f'    """\n    {description}\n\n    Args:'
    if docstring_params:
        docstring += "\n" + "\n".join(docstring_params)
    docstring += '\n\n    Returns:\n        dict: API response or error details\n    """'

    # Generate function code
    return f'''
@tool
def {name}({param_str}) -> dict:
{docstring}
    import requests
    import json
    from urllib.parse import urljoin
    from typing import Optional, Any

    # Prepare headers - combine defaults with environment headers
    headers = {json.dumps(DEFAULT_HEADERS, indent=4)}
    try:
        env_headers = json.loads(os.getenv("HEADERS", "{{}}"))
        headers.update(env_headers)
    except json.JSONDecodeError:
        pass

    # Build URL
    base_url = "{BASE_URL.rstrip('/')}"
    path = f"{url_path}"
    url = urljoin(base_url + "/", path)

    # Prepare request parameters
    params = {query_dict}
    json_data = {json_dict}

    try:
        response = requests.{method}(
            url,
            headers=headers,
            params=params,
            json=json_data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {{
            "error": "API request failed",
            "details": str(e),
            "status_code": getattr(e.response, "status_code", None),
            "response_text": getattr(e.response, "text", None)
        }}
    except Exception as e:
        return {{
            "error": "Unexpected error",
            "details": str(e)
        }}
'''.strip()

def generate_all_tools() -> None:
    """Generate all tool functions from OpenAPI specification."""
    if not os.path.exists(TOOLS_JSON):
        extract_tools_from_openapi()

    with open(TOOLS_JSON, "r") as f:
        tools = json.load(f)

    output = [
        '"""Auto-generated API tools from OpenAPI specification."""',
        "from langchain.tools import tool",
        "from typing import Any, Dict, Optional",
        "import os",
        "import json",
        "import requests",
        "from urllib.parse import urljoin",
        "",
        "# Default headers including required X-CLIENT-ID",
        f"DEFAULT_HEADERS = {json.dumps(DEFAULT_HEADERS, indent=4)}",
        ""
    ]

    for tool in tools:
        output.append(generate_tool_function(tool))
        output.append("")

    with open(OUTPUT_PY, "w") as f:
        f.write("\n".join(output))

    print(f"✅ Generated {len(tools)} functions in {OUTPUT_PY}")

if __name__ == "__main__":
    generate_all_tools()