from flask import Flask, request, jsonify, render_template
import os
import openai
import json
import logging
from typing import List, Dict, Any, Optional
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from langchain.tools.base import BaseTool
import vector_tools
import generated_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MAX_HISTORY_LENGTH = 20  # Prevent memory issues with long conversations
    DEFAULT_MODEL = "gpt-4o"
    TOP_K_TOOLS = 5

# Validate API key
if not Config.OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OpenAI API key is required")

openai.api_key = Config.OPENAI_API_KEY

class ChatbotManager:
    def __init__(self):
        self.history: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant with access to various tools. Use them when appropriate to provide accurate and helpful responses."}
        ]
        self.tool_functions: Dict[str, BaseTool] = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize tool functions from the generated_tools module"""
        try:
            if not hasattr(vector_tools, 'functions_schema'):
                logger.warning("vector_tools.functions_schema not found")
                return
                
            for func_spec in vector_tools.functions_schema:
                func_name = func_spec["name"]
                func = getattr(generated_tools, func_name, None)
                
                if func and isinstance(func, BaseTool):
                    self.tool_functions[func_name] = func
                    logger.info(f"Loaded tool: {func_name}")
                else:
                    logger.warning(f"Tool {func_name} not found or invalid in generated_tools")
                    
            logger.info(f"Initialized {len(self.tool_functions)} tools")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {str(e)}")
    
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
        func_name = tool_call.function.name
        raw_args = tool_call.function.arguments
        
        func = self.tool_functions.get(func_name)
        if not func:
            error_msg = f"Function {func_name} not found"
            logger.error(error_msg)
            return error_msg
        
        try:
            # Handle parameter-less functions specially
            if func_name == "get_license":
                args_dict = {}
            else:
                args_dict = self._safe_parse_args(raw_args)
            
            logger.info(f"Executing tool {func_name} with args: {args_dict}")
            
            # Call using modern invoke() if available
            if hasattr(func, 'invoke'):
                result = func.invoke(args_dict)
            else:
                # For __call__, convert to string if needed
                tool_input = json.dumps(args_dict) if args_dict else "{}"
                result = func(tool_input)
            
            # Format the result with error handling
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
            # Handle None or empty results
            if result is None:
                return "Operation completed successfully (no data returned)"
            
            # Handle string results that might be JSON
            if isinstance(result, str):
                # Handle empty strings
                if not result.strip():
                    return "Operation completed successfully (no data returned)"
                
                try:
                    parsed_result = json.loads(result)
                    return self._format_tool_result(parsed_result)
                except json.JSONDecodeError:
                    return result
            
            # Handle dictionary results
            if isinstance(result, dict):
                # Handle empty dictionaries
                if not result:
                    return "Operation completed successfully (no data returned)"
                
                # Check if it's an error response with error_description
                if "error" in result and "error_description" in result:
                    return result["error_description"]
                
                # Check for nested error_description in response_text
                if "response_text" in result and isinstance(result["response_text"], str):
                    try:
                        response_data = json.loads(result["response_text"])
                        if isinstance(response_data, dict) and "error_description" in response_data:
                            return response_data["error_description"]
                    except json.JSONDecodeError:
                        pass
                
                # Check for nested error_description in details
                if "details" in result and isinstance(result["details"], str):
                    try:
                        details = json.loads(result["details"])
                        if isinstance(details, dict) and "error_description" in details:
                            return details["error_description"]
                    except json.JSONDecodeError:
                        pass
                
                # Check if this is a general error response - return a user-friendly message
                if "error" in result and "status_code" in result:
                    error_msg = result.get("error", "An error occurred")
                    status_code = result.get("status_code")
                    
                    # Try to extract more specific error from response_text
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
                    
                    # Return a cleaned up error message
                    if status_code == 401:
                        return "Unauthorized access - please check your permissions or license"
                    elif status_code == 403:
                        return "Access forbidden - insufficient permissions"
                    elif status_code == 404:
                        return "Resource not found"
                    else:
                        return f"API Error: {error_msg}"
                
                # Check if this is a success response with status_code 200 but empty data
                if "status_code" in result and result["status_code"] == 200:
                    # Check if response_text is empty or contains empty data
                    response_text = result.get("response_text", "")
                    if not response_text or response_text.strip() in ["", "null", "[]", "{}"]:
                        return "Operation completed successfully (no data returned)"
                
                # Return formatted JSON for non-error responses
                return json.dumps(result, indent=2)
            
            # Handle list results
            if isinstance(result, list):
                # Handle empty lists
                if not result:
                    return "Operation completed successfully (no data returned)"
                return json.dumps(result, indent=2)
            
            # Return string representation for other types
            return str(result)
            
        except Exception as e:
            logger.warning(f"Error formatting tool result: {str(e)}")
            return str(result)
    
    def _manage_history_length(self):
        """Keep conversation history within reasonable limits"""
        if len(self.history) > Config.MAX_HISTORY_LENGTH:
            # Keep system message and recent messages
            system_msg = self.history[0]  # Keep system message
            recent_messages = self.history[-(Config.MAX_HISTORY_LENGTH-1):]
            self.history = [system_msg] + recent_messages
            logger.info("Conversation history trimmed")
        """Keep conversation history within reasonable limits"""
        if len(self.history) > Config.MAX_HISTORY_LENGTH:
            # Keep system message and recent messages
            system_msg = self.history[0]  # Keep system message
            recent_messages = self.history[-(Config.MAX_HISTORY_LENGTH-1):]
            self.history = [system_msg] + recent_messages
            logger.info("Conversation history trimmed")
    
    def process_message(self, user_message: str) -> str:
        """Process a user message and return the assistant's response"""
        try:
            # Add user message to history
            self.history.append({"role": "user", "content": user_message})
            self._manage_history_length()
            
            # Search for relevant tools
            try:
                top_tools = vector_tools.search_tools(user_message, top_k=Config.TOP_K_TOOLS)
                tools_spec = self._get_tools_spec_for_names(top_tools)
                logger.info(f"Found {len(tools_spec)} relevant tools")
            except Exception as e:
                logger.warning(f"Error searching tools: {str(e)}. Proceeding without tools.")
                tools_spec = []
            
            # Call OpenAI API
            api_params = {
                "model": Config.DEFAULT_MODEL,
                "messages": self.history,
            }
            
            if tools_spec:
                api_params["tools"] = tools_spec
                api_params["tool_choice"] = "auto"
            
            response = openai.chat.completions.create(**api_params)
            response_message = response.choices[0].message
            tool_calls = getattr(response_message, "tool_calls", None)
            
            if tool_calls:
                # Execute tool calls
                logger.info(f"Executing {len(tool_calls)} tool calls")
                
                # Add the assistant's message with tool calls to history first
                self.history.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        } for tc in tool_calls
                    ]
                })
                
                # Execute each tool call and add results to history
                for tool_call in tool_calls:
                    result = self._execute_tool_call(tool_call)
                    self.history.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id
                    })
                
                # Make a second API call to get the final response based on tool results
                final_response = openai.chat.completions.create(
                    model=Config.DEFAULT_MODEL,
                    messages=self.history
                )
                
                assistant_reply = final_response.choices[0].message.content or "I apologize, but I couldn't generate a response based on the tool results."
                self.history.append({"role": "assistant", "content": assistant_reply})
                
            else:
                # Regular response without tools
                assistant_reply = response_message.content or "I apologize, but I couldn't generate a response."
                self.history.append({"role": "assistant", "content": assistant_reply})
            
            return assistant_reply
            
        except openai.APIError as e:
            error_msg = f"OpenAI API error: {str(e)}"
            logger.error(error_msg)
            return "I'm experiencing technical difficulties. Please try again later."
            
        except Exception as e:
            error_msg = f"Unexpected error processing message: {str(e)}"
            logger.error(error_msg)
            return "An unexpected error occurred. Please try again."
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation"""
        return {
            "message_count": len(self.history) - 1,  # Exclude system message
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

@app.route("/")
def home():
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
        
        if len(user_message) > 4000:  # Reasonable message length limit
            return jsonify({"error": "Message too long"}), 400
        
        logger.info(f"Processing message: {user_message[:100]}...")
        
        # Process the message
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
    app.run(debug=True, host="0.0.0.0", port=8300)