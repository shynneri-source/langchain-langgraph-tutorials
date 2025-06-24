from langchain_google_genai import ChatGoogleGenerativeAI
# this import for use gemini model with api key

from dotenv import load_dotenv
import os
# these import for load environment variables from .env file

from pydantic import BaseModel, Field
# this import for use Pydantic to define the structure of the response



# Load environment variables from .env file
load_dotenv()

# initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    api_key = os.getenv("GOOGLE_API_KEY"),
)

# define a structured response model using Pydantic
# this is a way to define the structure of the response from the model
# you can use Pydantic's BaseModel to define the structure of the response
# and use Field to define the description of each field
class structured_response(BaseModel):
    """
    A structured response model for the Google Generative AI response.
    """
    user_input : str = Field(description="the user input for response")
    thinking : str = Field(description="think about the problem and provide for the user input")
    response : str = Field(description="the response from the model")

setup_structure = llm.with_structured_output(structured_response) #type: ignore[call-arg]

response = setup_structure.invoke("tell me something about math proofs")    #type: ignore[call-arg]

print(response) #type: ignore[no-untyped-call]

"""
output: 
user_input='tell me something about math proofs' 
thinking='The user asked a general question about math proofs, so I provided a brief overview of their purpose and some common types.' 
response='Math proofs are arguments that demonstrate a mathematical statement is true. They use logic and previously established facts to reach a conclusion. Different types of proofs exist, like direct proofs, proofs by contradiction, and proofs by induction.'
"""

