"""
LLM Provider Usage Example

This example demonstrates how to use LLM providers for text completion.
"""

import asyncio
from cogent_base.providers.completion import LiteLLMCompletionModel

async def main():
    # Initialize the model
    model = LiteLLMCompletionModel("gpt-4")
    
    # Create a simple completion request
    # Note: You'll need to implement the proper request structure
    # based on your completion model's interface
    
    print("LLM Provider initialized successfully!")
    print(f"Model: {model}")
    
    # Example usage would depend on your specific request format
    # response = await model.complete(request)

if __name__ == "__main__":
    asyncio.run(main())