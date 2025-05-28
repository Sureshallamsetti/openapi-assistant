from flask import Flask, request, jsonify, render_template, Blueprint, redirect, url_for, flash
import os
import requests
import json
import logging
from typing import List, Dict, Any, Optional
from langchain.tools.base import BaseTool
import vector_tools
from tools import generated_tools
from dotenv import load_dotenv
import yaml
import subprocess
import tempfile
from werkzeug.utils import secure_filename
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

load_dotenv()

# Configuration
class Config:
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1")
    MAX_HISTORY_LENGTH = 20
    DEFAULT_MODEL = "mistral-nemo:12b-instruct-2407-q4_0"
    TOP_K_TOOLS = 5
    UPLOAD_FOLDER = 'uploads'
    EXISTING_SPECS_FOLDER = 'api_specs'
    ALLOWED_EXTENSIONS = {'yaml', 'yml', 'json'}

# Create directories if they don't exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.EXISTING_SPECS_FOLDER, exist_ok=True)
os.makedirs('tools', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def get_existing_specs():
    """Get list of existing API specification files"""
    specs = []
    if os.path.exists(Config.EXISTING_SPECS_FOLDER):
        for filename in os.listdir(Config.EXISTING_SPECS_FOLDER):
            if allowed_file(filename):
                filepath = os.path.join(Config.EXISTING_SPECS_FOLDER, filename)
                specs.append({
                    'filename': filename,
                    'name': filename.rsplit('.', 1)[0],
                    'size': os.path.getsize(filepath),
                    'modified': os.path.getmtime(filepath)
                })
    return sorted(specs, key=lambda x: x['modified'], reverse=True)

def run_tools_generator(input_file_path):
    """Run the tools generator script on the given file"""
    try:
        # Import the tools generator functions
        import sys
        sys.path.append('.')
        
        # Import the main function from tools_generator
        from tools_generator import extract_tools_from_openapi, generate_tools_from_json
        
        logger.info(f"Processing API spec: {input_file_path}")
        
        if input_file_path.endswith(('.yaml', '.yml')):
            # OpenAPI workflow
            tools = extract_tools_from_openapi(input_file_path)
            
            # Save intermediate tools.json
            tools_json_path = "tools.json"
            with open(tools_json_path, "w") as f:
                json.dump(tools, f, indent=2)
            
            # Generate Python tools and schema
            generate_tools_from_json(tools_json_path)
            
        elif input_file_path.endswith('.json'):
            # JSON workflow
            generate_tools_from_json(input_file_path)
        
        logger.info("Tools generation completed successfully")
        return True, "Tools generated successfully"
        
    except Exception as e:
        error_msg = f"Error generating tools: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

class ChatbotManager:
    def __init__(self):
        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant with access to various tools. Use them when appropriate to provide accurate and helpful responses."}
        ]
        self.tool_functions: Dict[str, BaseTool] = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize tool functions from the generated_tools module"""
        try:
            # # Reload the generated_tools module to pick up new tools
            # import importlib,sys
            # if 'tools.generated_tools' in sys.modules:
            #     importlib.reload(sys.modules['tools.generated_tools'])
            
            # from tools import generated_tools
            
            if not hasattr(vector_tools, 'functions_schema'):
                logger.warning("vector_tools.functions_schema not found")
                return
                
            for func_spec in vector_tools.functions_schema:
                func_name = func_spec["name"]
                func = getattr(generated_tools, func_name, None)
                
                if func and isinstance(func, BaseTool):
                    self.tool_functions[func_name] = func
                    # logger.info(f"Loaded tool: {func_name}")
                else:
                    logger.warning(f"Tool {func_name} not found or invalid in generated_tools")
                    
            logger.info(f"Initialized {len(self.tool_functions)} tools")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {str(e)}")
    
    def reload_tools(self):
        """Reload tools after new tools are generated"""
        self.tool_functions.clear()
        self._initialize_tools()
    
    def _get_tools_spec_for_names(self, tools_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build tools spec JSON only for selected tool names"""
        return [{"type": "function", "function": tool_spec} for tool_spec in tools_list]
    
    def _safe_parse_args(self, raw_args: str) -> Dict[str, Any]:
        """Handle empty/malformed arguments for parameter-less functions"""
        if not raw_args or raw_args.strip() in ['', 'null', '{}', 'None']:
            return {}
        
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool arguments: {raw_args}. Error: {str(e)}")
            return {}
    
    def _execute_tool_call(self, tool_call) -> str:
        """Execute a single tool call and return the result"""
        func_name = tool_call["function"]["name"]
        raw_args = tool_call["function"]["arguments"]
        
        func = self.tool_functions.get(func_name)
        if not func:
            error_msg = f"Function {func_name} not found"
            logger.error(error_msg)
            return error_msg
        
        try:
            args_dict = self._safe_parse_args(raw_args)
            logger.info(f"Executing tool {func_name} with args: {args_dict}")
            
            # Call using modern invoke() if available
            if hasattr(func, 'invoke'):
                result = func.invoke(args_dict)
            else:
                tool_input = json.dumps(args_dict) if args_dict else "{}"
                result = func(tool_input)
            
            formatted_result = self._format_tool_result(result)
            logger.info(f"Tool {func_name} executed successfully")
            return formatted_result
            
        except Exception as e:
            error_msg = f"Error calling {func_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _format_tool_result(self, result) -> str:
        """Format tool result, extracting error_description from error responses"""
        try:
            if result is None:
                return "Operation completed successfully (no data returned)"
            
            if isinstance(result, str):
                if not result.strip():
                    return "Operation completed successfully (no data returned)"
                
                try:
                    parsed_result = json.loads(result)
                    return self._format_tool_result(parsed_result)
                except json.JSONDecodeError:
                    return result
            
            if isinstance(result, dict):
                if not result:
                    return "Operation completed successfully (no data returned)"
                
                if "error" in result and "error_description" in result:
                    return result["error_description"]
                
                if "response_text" in result and isinstance(result["response_text"], str):
                    try:
                        response_data = json.loads(result["response_text"])
                        if isinstance(response_data, dict) and "error_description" in response_data:
                            return response_data["error_description"]
                    except json.JSONDecodeError:
                        pass
                
                if "details" in result and isinstance(result["details"], str):
                    try:
                        details = json.loads(result["details"])
                        if isinstance(details, dict) and "error_description" in details:
                            return details["error_description"]
                    except json.JSONDecodeError:
                        pass
                
                if "error" in result and "status_code" in result:
                    error_msg = result.get("error", "An error occurred")
                    status_code = result.get("status_code")
                    
                    if "response_text" in result:
                        try:
                            response_data = json.loads(result["response_text"])
                            if isinstance(response_data, dict):
                                if "error_description" in response_data:
                                    return response_data["error_description"]
                                elif "error" in response_data:
                                    return response_data["error"]
                        except json.JSONDecodeError:
                            pass
                    
                    if status_code == 401:
                        return "Unauthorized access - please check your permissions or license"
                    elif status_code == 403:
                        return "Access forbidden - insufficient permissions"
                    elif status_code == 404:
                        return "Resource not found"
                    else:
                        return f"API Error: {error_msg}"
                
                if "status_code" in result and result["status_code"] == 200:
                    response_text = result.get("response_text", "")
                    if not response_text or response_text.strip() in ["", "null", "[]", "{}"]:
                        return "Operation completed successfully (no data returned)"
                
                return json.dumps(result, indent=2)
            
            if isinstance(result, list):
                if not result:
                    return "Operation completed successfully (no data returned)"
                return json.dumps(result, indent=2)
            
            return str(result)
            
        except Exception as e:
            logger.warning(f"Error formatting tool result: {str(e)}")
            return str(result)
    
    def _manage_history_length(self):
        """Keep conversation history within reasonable limits"""
        if len(self.history) > Config.MAX_HISTORY_LENGTH:
            system_msg = self.history[0]
            recent_messages = self.history[-(Config.MAX_HISTORY_LENGTH-1):]
            self.history = [system_msg] + recent_messages
            logger.info("Conversation history trimmed")

    def process_message(self, user_message: str) -> str:
        """Process a user message with minimal history to reduce hallucination"""
        try:
            messages = [{"role": "user", "content": user_message}]

            tools_spec = []
            try:
                top_tools = vector_tools.search_tools(user_message, top_k=Config.TOP_K_TOOLS)
                tools_spec = self._get_tools_spec_for_names(top_tools)
                logger.info(f"Found {len(tools_spec)} relevant tools")
            except Exception as e:
                logger.warning(f"Error searching tools: {str(e)}. Proceeding without tools.")

            api_params = {
                "model": Config.DEFAULT_MODEL,
                "messages": messages,
                "temperature": 0.1,
            }

            if tools_spec:
                api_params["tools"] = tools_spec

            response = requests.post(
                f"{Config.OLLAMA_API_URL}/chat/completions",
                json=api_params,
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()

            response_message = response_data["choices"][0]["message"]
            tool_calls = response_message.get("tool_calls")

            if tool_calls:
                logger.info(f"Executing {len(tool_calls)} tool calls")
                
                tool_results = []
                for tool_call in tool_calls:
                    try:
                        result = self._execute_tool_call(tool_call)
                        tool_name = tool_call.get("function", {}).get("name", "Unknown")
                        tool_results.append({
                            "tool": tool_name,
                            "result": str(result)[:1000]
                        })
                    except Exception as e:
                        logger.error(f"Tool execution error: {str(e)}")
                        tool_results.append({
                            "tool": tool_call.get("function", {}).get("name", "Unknown"),
                            "result": f"Error: {str(e)}"
                        })

                final_messages = [
                    {
                        "role": "user",
                        "content": f"Question: {user_message}\n\nTool Results: {tool_results}\n\nProvide a direct answer based only on the tool results above."
                    }
                ]

                final_response = requests.post(
                    f"{Config.OLLAMA_API_URL}/chat/completions",
                    json={
                        "model": Config.DEFAULT_MODEL,
                        "messages": final_messages,
                        "temperature": 0.1,
                    },
                    timeout=30
                )
                final_response.raise_for_status()
                final_data = final_response.json()
                
                return final_data["choices"][0]["message"]["content"]

            else:
                return response_message.get("content", "I couldn't generate a response.")

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return "I encountered an error processing your request."
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation"""
        return {
            "message_count": len(self.history) - 1,
            "available_tools": len(self.tool_functions),
            "tool_names": list(self.tool_functions.keys())
        }
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.history = [
            {"role": "system", "content": "You are a helpful assistant with access to various tools. Use them when appropriate to provide accurate and helpful responses."}
        ]
        logger.info("Conversation reset")

# Global chatbot manager instance
chatbot = ChatbotManager()

# Welcome page routes
@app.route("/")
def welcome():
    """Serve the welcome page"""
    existing_specs = get_existing_specs()
    return render_template("welcome.html", existing_specs=existing_specs)

@app.route("/upload-api", methods=["POST"])
def upload_file():
    """Handle file upload of API spec and regenerate tools (AJAX-compatible)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file.filename and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)

        try:
            file.save(filepath)
            logger.info(f"Uploaded file saved: {filepath}")

            success, message = run_tools_generator(filepath)
            if success:
                chatbot.reload_tools()
                return jsonify({
                    'message': 'API spec uploaded and tools generated successfully.',
                    'redirect': url_for('chat_interface')
                }), 200
            else:
                return jsonify({'error': f'Failed to generate tools: {message}'}), 500

        except Exception as e:
            logger.error(f"Error saving or processing file: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type. Allowed types: yaml, yml, json'}), 400

@app.route("/list-apis", methods=["GET"])
def list_apis():
    try:
        files = os.listdir(Config.EXISTING_SPECS_FOLDER)
    except Exception as e:
        return jsonify({"error": str(e), "apis": []}), 500

    apis = []
    for f in files:
        if f.lower().endswith(('.yaml', '.yml', '.json')):  # filter for valid spec files
            apis.append({
                "id": f,  # You can use the filename as ID
                "name": f,
                "uploaded_at": "unknown",  # TODO: add actual upload time if tracked
                "endpoints": 0,             # TODO: count endpoints from spec if desired
            })
    return jsonify({"apis": apis})

@app.route("/select-api", methods=["POST"])
def select_existing():
    selected_file = request.form.get('selected_file')
    if not selected_file:
        flash('No file selected')
        return redirect(url_for('welcome'))
    
    filepath = os.path.join(Config.EXISTING_SPECS_FOLDER, selected_file)
    if not os.path.exists(filepath):
        flash('Selected file not found')
        return redirect(url_for('welcome'))
    
    success, message = run_tools_generator(filepath)
    
    if success:
        chatbot.reload_tools()
        flash(f'API specification loaded and tools generated successfully!')
        return redirect(url_for('chat_interface'))
    else:
        flash(f'Error generating tools: {message}')
        return redirect(url_for('welcome'))

@app.route("/chat")
def chat_interface():
    """Serve the main chat interface"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat messages"""
    if not request.is_json:
        return jsonify({"error": "Invalid request, JSON expected"}), 400

    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        if not isinstance(user_message, str):
            return jsonify({"error": "Message must be a string"}), 400
        
        if len(user_message) > 4000:
            return jsonify({"error": "Message too long"}), 400
        
        logger.info(f"Processing message: {user_message[:100]}...")
        
        reply = chatbot.process_message(user_message)
        
        return jsonify({
            "reply": reply,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": "An error occurred processing your request",
            "status": "error"
        }), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get conversation and system statistics"""
    try:
        stats = chatbot.get_conversation_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({"error": "Could not retrieve stats"}), 500

@app.route("/reset", methods=["POST"])
def reset_conversation():
    """Reset the conversation history"""
    try:
        chatbot.reset_conversation()
        return jsonify({"message": "Conversation reset successfully", "status": "success"})
    except Exception as e:
        logger.error(f"Error resetting conversation: {str(e)}")
        return jsonify({"error": "Could not reset conversation"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "tools_loaded": len(chatbot.tool_functions) > 0
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting Flask chatbot application...")
    logger.info(f"Loaded {len(chatbot.tool_functions)} tools")
    app.run(debug=False, host="0.0.0.0", port=8300)