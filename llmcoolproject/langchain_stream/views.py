from langchain.llms.ollama import Ollama
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from channels.generic.websocket import AsyncWebsocketConsumer
import json
from dotenv import load_dotenv
import os

load_dotenv(".env")

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("user", "{input}")]
)

# Initialize both models
try:
    ollama_llm = Ollama(
        model="llama3.2",
        base_url="http://localhost:11434",
    )
except Exception as e:
    print(f"Error initializing Ollama: {e}")
    ollama_llm = None

try:
    openai_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True,
    )
except Exception as e:
    print(f"Error initializing ChatGPT: {e}")
    openai_llm = None

output_parser = StrOutputParser()

# Create chains for both models
ollama_chain = (
    (
        prompt
        | ollama_llm.with_config({"run_name": "model"})
        | output_parser.with_config({"run_name": "Assistant"})
    )
    if ollama_llm
    else None
)

openai_chain = (
    (
        prompt
        | openai_llm.with_config({"run_name": "model"})
        | output_parser.with_config({"run_name": "Assistant"})
    )
    if openai_llm
    else None
)


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        try:
            text_data_json = json.loads(text_data)
            message = text_data_json["message"]
            model_choice = text_data_json.get(
                "model", "ollama"
            )  # Default to ollama if not specified

            # Select the appropriate chain based on model_choice
            selected_chain = ollama_chain if model_choice == "ollama" else openai_chain

            if not selected_chain:
                error_message = f"Selected model '{model_choice}' is not available"
                await self.send(
                    text_data=json.dumps(
                        {"event": "error", "data": {"error": error_message}}
                    )
                )
                return

            try:
                async for chunk in selected_chain.astream_events(
                    {"input": message}, version="v1", include_names=["Assistant"]
                ):
                    if chunk["event"] in ["on_parser_start", "on_parser_stream"]:
                        await self.send(text_data=json.dumps(chunk))

            except Exception as e:
                error_message = f"Error processing message: {str(e)}"
                print(error_message)
                await self.send(
                    text_data=json.dumps(
                        {"event": "error", "data": {"error": error_message}}
                    )
                )

        except json.JSONDecodeError as e:
            print(f"Invalid JSON received: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
