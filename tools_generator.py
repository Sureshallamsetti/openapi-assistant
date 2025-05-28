
import json
import keyword
import os
import sys
import ast
from dotenv import load_dotenv
import yaml
import re
from typing import Dict, List, Optional, Any

"""
API Tools Generator
Generates Python tools from OpenAPI spec or tools.json and extracts JSON schema.

Usage:
    python tools_generator.py api.yaml      # From OpenAPI YAML
    python tools_generator.py tools.json    # From tools JSON
"""

def resolve_ref(ref, components):
    """Resolve $ref like '#/components/schemas/Foo'"""
    if not ref.startswith("#/"):
        return {}
    parts = ref.strip("#/").split("/")
    obj = components
    for part in parts[1:]:  # Skip 'components'
        obj = obj.get(part)
        if obj is None:
            return {}
    return obj

def parse_schema(schema, components):
    """Parse a schema object, resolving $ref if needed"""
    if "$ref" in schema:
        schema = resolve_ref(schema["$ref"], components)

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    tool_props = {}
    for prop_name, prop_schema in properties.items():
        if "$ref" in prop_schema:
            prop_schema = resolve_ref(prop_schema["$ref"], components)

        tool_props[prop_name] = {
            "type": prop_schema.get("type", "string"),
            "description": prop_schema.get("description", ""),
            "default": prop_schema.get("default", ""),
            "in": "body"
        }

    return {
        "type": "object",
        "properties": tool_props,
        "required": required
    }

def openapi_to_tools(openapi_spec):
    """Convert OpenAPI spec to tools format"""
    tools = []
    components = openapi_spec.get("components", {})

    for path, methods in openapi_spec.get("paths", {}).items():
        for method, operation in methods.items():
            if not isinstance(operation, dict) or method.lower() not in ["get", "post", "put", "delete", "patch"]:
                continue

            operation_id = operation.get("operationId")
            if not operation_id:
                # fallback: generate from method + path
                clean_path = path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
                operation_id = f"{method}_{clean_path}"
                
            description = operation.get("description", f"{method.upper()} {path}")
            tool_params = {
                "type": "object",
                "properties": {},
                "required": []
            }

            # Handle parameters
            for param in operation.get("parameters", []):
                if "$ref" in param:
                    param = resolve_ref(param["$ref"], components)

                name = param.get("name")
                schema = param.get("schema", {})
                if "$ref" in schema:
                    schema = resolve_ref(schema["$ref"], components)

                tool_params["properties"][name] = {
                    "type": schema.get("type", "string"),  
                    "description": param.get("description", ""),
                    "default": param.get("default", ""),
                    "in": param.get("in", "query")
                }

                if param.get("required", False):
                    tool_params["required"].append(name)

            # Handle request body
            request_body = operation.get("requestBody")
            if request_body:
                if "$ref" in request_body:
                    request_body = resolve_ref(request_body["$ref"], components)

                content = request_body.get("content", {})
                json_schema = content.get("application/json", {}).get("schema")
                if json_schema:
                    parsed = parse_schema(json_schema, components)
                    tool_params["properties"].update(parsed["properties"])
                    tool_params["required"] += parsed.get("required", [])

            # Remove duplicate required fields
            tool_params["required"] = list(set(tool_params["required"]))

            tools.append({
                "name": operation_id,
                "description": description,
                "method": method.lower(),  # Add HTTP method
                "path": path,            # Add endpoint path
                "parameters": tool_params
            })

    return tools

def extract_tools_from_openapi(openapi_path: str) -> List[Dict]:
    """Extract API operations from OpenAPI specification."""
    print(f"📖 Reading OpenAPI specification from {openapi_path}")
    
    with open(openapi_path, 'r') as f:
        spec = yaml.safe_load(f)
    
    tools = openapi_to_tools(spec)
    print(f"✅ Extracted {len(tools)} tools from OpenAPI specification")
    return tools#!/usr/bin/env python3

# Load environment variables
load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://localhost:8080/v3")
DEFAULT_HEADERS = json.loads(os.getenv("HEADERS", '{"X-CLIENT-ID": "1", "Content-Type": "application/json"}'))

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
            default_value = schema.get("default")
            if default_value is not None and default_value != "":
                # Properly format default values for different types
                if isinstance(default_value, str):
                    default_str = f'"{default_value}"'
                elif isinstance(default_value, bool):
                    default_str = str(default_value)  # True / False
                else:
                    default_str = str(default_value)
                optional_params.append(f"{safe_name}: Optional[{param_type}] = {default_str}")
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
            timeout=10,
            verify=False
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

def generate_tools_from_json(tools_json_path: str) -> None:
    """Generate Python tools from JSON specification."""
    print(f"📖 Reading tools from {tools_json_path}")
    
    with open(tools_json_path, "r") as f:
        tools = json.load(f)

    os.makedirs("tools", exist_ok=True)
    output_py = os.path.join("tools", "generated_tools.py")
    output = [
        '"""Auto-generated API tools from tools.json specification."""',
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

    with open(output_py, "w") as f:
        f.write("\n".join(output))

    print(f"✅ Generated {len(tools)} functions in {output_py}")
    
    # Extract schema from generated Python
    schema = extract_schema_from_python(output_py)
    
    # Save schema JSON
    os.makedirs("tools", exist_ok=True)
    output_schema =  os.path.join("tools","functions_schema.json")
    with open(output_schema, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"✅ Saved functions schema to {output_schema}")

# Schema extraction functions
def parse_default(value):
    """Handle default value types for JSON schema defaults."""
    if isinstance(value, ast.Constant):
        return value.value
    return None

def get_annotation(annotation) -> Optional[str]:
    """Extract type annotation as string."""
    if isinstance(annotation, ast.Name):
        return annotation.id  # e.g., 'str', 'bool', 'Any'
    elif isinstance(annotation, ast.Subscript):
        if isinstance(annotation.value, ast.Name) and annotation.value.id == "Optional":
            if isinstance(annotation.slice, ast.Name):
                return f"Optional[{annotation.slice.id}]"
            elif isinstance(annotation.slice, ast.Subscript):
                base_type = get_annotation(annotation.slice)
                return f"Optional[{base_type}]"
        elif isinstance(annotation.value, ast.Name):
            return annotation.value.id  # Handle List, Dict, etc.
    elif isinstance(annotation, ast.Attribute):
        return annotation.attr
    return None

def get_type_schema(py_type: Optional[str]):
    """Map Python types to JSON Schema types."""
    mapping = {
        "str": {"type": "string"},
        "int": {"type": "integer"},
        "float": {"type": "number"},
        "bool": {"type": "boolean"},
        "dict": {"type": "object"},
        "list": {"type": "array"},
        "Any": {"type": "string"},
        "Optional[str]": {"type": ["string", "null"]},
        "Optional[int]": {"type": ["integer", "null"]},
        "Optional[float]": {"type": ["number", "null"]},
        "Optional[bool]": {"type": ["boolean", "null"]},
        "Optional[dict]": {"type": ["object", "null"]},
        "Optional[list]": {"type": ["array", "null"]},
    }
    if py_type is None:
        return {"type": "string"}
    return mapping.get(py_type, {"type": "string"})

def extract_schema_from_python(filename: str) -> List[Dict]:
    """Extract tool schema from Python file with @tool decorated functions."""
    print(f"📖 Extracting schema from {filename}")
    
    with open(filename, "r") as f:
        source = f.read()
    tree = ast.parse(source)

    tools = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # Check if @tool decorator present
            decorators = [d.id if isinstance(d, ast.Name) else None for d in node.decorator_list]
            if "tool" in decorators:
                func_name = node.name
                docstring = ast.get_docstring(node) or ""
                # Extract only the first line of the docstring (the endpoint)
                description = docstring.strip().split('\n')[0].strip()

                # Build properties dict
                properties = {}
                required = []

                # Extract parameters with types and defaults
                args_with_defaults = []
                num_defaults = len(node.args.defaults)
                num_args = len(node.args.args)
                
                for i, arg in enumerate(node.args.args):
                    default_index = i - (num_args - num_defaults)
                    default = node.args.defaults[default_index] if default_index >= 0 else None
                    args_with_defaults.append((arg, default))

                for arg, default in args_with_defaults:
                    arg_name = arg.arg
                    annotation = None
                    if arg.annotation:
                        annotation = get_annotation(arg.annotation)

                    schema_type = get_type_schema(annotation)
                    param_info = schema_type.copy()
                    param_info["description"] = f"Parameter {arg_name} of type {annotation or 'string'}"

                    if default is not None:  # Check if default exists, including falsy values
                        default_val = parse_default(default)
                        if default_val is not None:
                            param_info["default"] = default_val  # Use raw value
                    else:
                        required.append(arg_name)

                    properties[arg_name] = param_info

                tool_schema = {
                    "name": func_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                }
                tools.append(tool_schema)

    print(f"✅ Extracted schema for {len(tools)} tools")
    return tools

def main():
    """Main function to process input file and generate Python tools + schema."""
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        print("  input_file can be:")
        print("    - api.yaml (OpenAPI specification)")
        print("    - tools.json (tools specification)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    print(f"🚀 Processing {input_file}")
    
    try:
        if input_file.endswith(('.yaml', '.yml')):
            # OpenAPI workflow: YAML -> tools.json -> Python -> schema JSON
            print("📋 Workflow: OpenAPI YAML → tools JSON → Python → schema JSON")
            
            # Extract tools from OpenAPI
            tools = extract_tools_from_openapi(input_file)
            
            # Save intermediate tools.json
            tools_json_path = "tools.json"
            with open(tools_json_path, "w") as f:
                json.dump(tools, f, indent=2)
            print(f"💾 Saved intermediate tools to {tools_json_path}")
            
            # Generate Python tools and schema
            generate_tools_from_json(tools_json_path)
            
        elif input_file.endswith('.json'):
            # JSON workflow: tools.json -> Python -> schema JSON
            print("📋 Workflow: tools JSON → Python → schema JSON")
            generate_tools_from_json(input_file)
            
        else:
            print(f"❌ Error: Unsupported file type '{input_file}'")
            print("Supported types: .yaml, .yml, .json")
            sys.exit(1)
        
        print("🎉 All done!")
        print("📁 Generated files:")
        print("   - generated_tools.py (Python tools)")
        print("   - functions_schema.json (JSON schema)")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
