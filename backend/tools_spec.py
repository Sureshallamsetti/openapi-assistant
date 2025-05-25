import ast
import inspect
import json
from typing import get_type_hints, Optional

def parse_default(value):
    # Handle some common default value types for JSON schema defaults
    if isinstance(value, ast.Constant):
        return value.value
    if isinstance(value, ast.NameConstant):
        return value.value
    if isinstance(value, ast.Str):
        return value.s
    if isinstance(value, ast.Num):
        return value.n
    return None

def get_annotation(annotation) -> Optional[str]:
    # Return annotation name as string (simple version)
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Subscript):
        if isinstance(annotation.value, ast.Name):
            return annotation.value.id
        else:
            return None
    elif isinstance(annotation, ast.Attribute):
        return annotation.attr
    return None

def get_type_schema(py_type: Optional[str]):
    # Map simple Python types to JSON Schema types
    mapping = {
        "str": {"type": "string"},
        "int": {"type": "integer"},
        "float": {"type": "number"},
        "bool": {"type": "boolean"},
        "dict": {"type": "object"},
        "list": {"type": "array"},
        "Optional[str]": {"type": ["string", "null"]},
        "Optional[int]": {"type": ["integer", "null"]},
        # Add more mappings if needed
    }
    if py_type is None:
        return {"type": "string"}  # default fallback
    return mapping.get(py_type, {"type": "string"})

def extract_tools_from_file(filename):
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

                # Build properties dict
                properties = {}
                required = []

                # Extract parameters with types and defaults
                for arg, default in list(zip(
                    node.args.args[::-1],
                    (node.args.defaults + [None] * (len(node.args.args) - len(node.args.defaults)))[::-1]
                ))[::-1]:
                # rest of your code


                    arg_name = arg.arg
                    annotation = None
                    if arg.annotation:
                        annotation = get_annotation(arg.annotation)

                    schema_type = get_type_schema(annotation)
                    param_info = schema_type.copy()
                    param_info["description"] = f"Parameter {arg_name} of type {annotation or 'string'}"

                    if default:
                        default_val = parse_default(default)
                        if default_val is not None:
                            param_info["default"] = str(default_val)
                    else:
                        required.append(arg_name)

                    properties[arg_name] = param_info

                tool_schema = {
                    "name": func_name,
                    "description": docstring.strip(),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                }
                tools.append(tool_schema)

    return tools

if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else "generated_tools.py"
    tools_schema = extract_tools_from_file(filename)
    print(json.dumps(tools_schema, indent=2))
