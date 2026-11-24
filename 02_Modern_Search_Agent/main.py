from typing import List

# Load environment variables (like TAVILY_API_KEY and LANGSMITH_API_KEY) from the .env file
from dotenv import load_dotenv
load_dotenv()

# Pydantic is used to define strict data schemas for the agent's final output
from pydantic import BaseModel, Field

# LangChain imports for creating the agent, formatting messages, LLM, and search tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


# ---------------------------------------------------------
# 1. DEFINE OUTPUT SCHEMAS (Structured Output)
# ---------------------------------------------------------
class Source(BaseModel):
    """Schema representing a single web source URL used by the agent."""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema forcing the agent to return a structured answer alongside its sources."""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer"
    )


# ---------------------------------------------------------
# 2. INITIALIZE BRAIN (LLM), TOOLS, AND AGENT
# ---------------------------------------------------------
# Connect to the local Llama 3.2 model running via Ollama
llm = ChatOllama(model="llama3.2:latest")

# Give the agent access to Tavily so it can search the live web
tools = [TavilySearch()]

# Combine the LLM, tools, and output schema into an autonomous agent loop
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse
)


# ---------------------------------------------------------
# 3. RUN THE AGENT
# ---------------------------------------------------------
def main():
    print("Hello from langchain-course!")

    # Invoke the agent by sending a user query as a HumanMessage
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the latest news about the Mars Rover Perseverance?"
                )
            ]
        }
    )

    # Print the full state dictionary returned by the agent
    print(result)

    # Tip: To print only the clean structured output, uncomment the lines below:
    # Extract the structured Pydantic object returned by the agent
    structured = result.get("structured_response")

    if structured:
        print("\n=== FINAL ANSWER ===")
        print(structured.answer)

        print("\n=== SOURCES ===")
        for source in structured.sources:
            print("-", source.url)
    else:
        # Fallback: print the last message content if structured_response wasn't populated
        print("\n=== FINAL MESSAGE ===")
        print(result["messages"][-1].content)



# Standard Python entry point: runs main() only when this file is executed directly
if __name__ == "__main__":
    main()
