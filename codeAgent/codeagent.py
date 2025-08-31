# CodeAgent: The main agent class that coordinates tools and models to perform tasks.
# DuckDuckGoSearchTool: A tool that allows the agent to search the web using DuckDuckGo.
# InferenceClientModel: A model wrapper that connects to Hugging Face's inference API using your token.

from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel
import os

# Read Hugging Face token from environment variable
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()], model=InferenceClientModel(token=HUGGINGFACE_TOKEN)
)

agent.run(
    "Search for the best music recommendations for a party at the Wayne's mansion."
)
