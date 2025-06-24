#in this file we will use the streaming response from the model
#because gemini model support streaming response, we can use another model to use the streaming response
#In this way I choose qwen3_4b from lmstudio to use the streaming response
#but if you want to use streaming response , i recommend you use langgraph library
#because it is a powerful library for building stateful workflows with streaming responses
#so you can follow this way to use the streaming response from the model

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages


class MessageState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(
    model="qwen/qwen3-4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="not-needed-for-lmstudio", # API key is not needed for local LM Studio
    streaming=True,  # Enable streaming response
)

def call_llm(messages: list[MessageState]) -> MessageState:
    """
    Call the LLM with the given messages and return the response.
    """
    response = llm.invoke(messages["messages"])  # type: ignore[call-arg]
    return response  # type: ignore[no-untyped-return]


workflow = StateGraph(MessageState)  # type: ignore[no-untyped-call]

workflow.add_node(
    "call_llm",
    call_llm
)

workflow.set_entry_point("call_llm")

graph = workflow.compile()

data = {
    "messages": [
        SystemMessage(content="You are a helpful assistant that provides information about math proofs."),
        HumanMessage(content="tell me something about math proofs")
    ]
}

for chunk, metadat in graph.stream(data, stream_mode="messages"):
    print(chunk.content, end="", flush=True)  # type: ignore[no-untyped-call])


#so with that way you can get a streaming response from the model
#so you can use this feature to build more interactive applications
#so the basic for how can get structure response from the model is done you want more , you can explore the langchain documentation and the langgraph documentation
