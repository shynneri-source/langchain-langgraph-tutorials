from langchain_google_genai import ChatGoogleGenerativeAI
#this import for use gemini model with api key

from typing import TypedDict, Annotated
#these import for use TypedDict and Annotated to define the structure of the response 

from dotenv import load_dotenv
import os
# this import for load environment variables from .env file

# Load environment variables from .env file
load_dotenv()

# initialize the Google Generative AI model 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)


# define a structured response model using TypedDict
# this is a way to define the structure of the response from the model
class structured_response(TypedDict):
    """
    A structured response model for the Google Generative AI response.
    """
    user_input: Annotated[str, "the user input for response"]
    thinking: Annotated[str, "think about the problem and provide for the user input"]
    response: Annotated[str, "the response from the model"]

structure_response = llm.with_structured_output(structured_response)  # type: ignore[call-arg]
response = structure_response.invoke("tell me something about math proofs")  # type: ignore[call-arg]
print(response)  # type: ignore[no-untyped-call]

print(response.keys()) # type: ignore[no-untyped-call]

"""
Output:
{'user_input': 'tell me something about math proofs',
'response': 'Math proofs are logical arguments that demonstrate the truth of a mathematical statement. They start with axioms (basic assumptions) and use deductive reasoning to reach a conclusion. Different proof techniques exist, like direct proof, proof by contradiction, and proof by induction.',
'thinking': 'The user asked a general question about math proofs, so I will provide a brief overview of what they are.'}
"""

# but if you want another ordinal output, you can follow this way:
ordered_keys = ["user_input", "thinking", "response"]
ordered_response = {key: response[key] for key in ordered_keys} # type: ignore[no-untyped-call]
print(ordered_response) # type: ignore[no-untyped-call]

"""
so after that you can get the ordered output like this:
Output:
{'user_input': 'tell me something about math proofs',
 'thinking': 'The user asked a general question about math proofs, so I will provide a brief overview of what they are.',
 'response': 'Math proofs are logical arguments that demonstrate the truth of a mathematical statement. They start with axioms (basic assumptions) and use deductive reasoning to reach a conclusion. Different proof techniques exist, like direct proof, proof by contradiction, and proof by induction.'}
"""

