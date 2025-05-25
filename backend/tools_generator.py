import yaml
import json

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
            "description": prop_schema.get("description", "")
        }

    return {
        "type": "object",
        "properties": tool_props,
        "required": required
    }

def openapi_to_tools(openapi_spec):
    tools = []
    components = openapi_spec.get("components", {})

    for path, methods in openapi_spec.get("paths", {}).items():
        for method, operation in methods.items():
            if not isinstance(operation, dict) or method.lower() not in ["get", "post", "put", "delete", "patch"]:
                continue

            operation_id = operation.get("operationId")
            if not operation_id:
                continue

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
                    "description": param.get("description", "")
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

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) != 2:
        print("Usage: python generate_tools_from_openapi.py <openapi_file.yaml>")
        sys.exit(1)

    openapi_file = sys.argv[1]

    if not os.path.isfile(openapi_file):
        print(f"File not found: {openapi_file}")
        sys.exit(1)

    with open(openapi_file, "r") as f:
        print("Loading OpenAPI spec...")
        spec = yaml.safe_load(f)

    print("Generating tools...")
    tools = openapi_to_tools(spec)

    output_file = "tools.json"
    with open(output_file, "w") as f:
        json.dump(tools, f, indent=2)

    print(f"✅ Tool schema written to {output_file}")
