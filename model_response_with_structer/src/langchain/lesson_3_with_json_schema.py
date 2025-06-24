from langchain_google_genai import ChatGoogleGenerativeAI
# this import for use gemini model with api key

from dotenv import load_dotenv
import os
# these import for load environment variables from .env file

# Load environment variables from .env file
load_dotenv()

# initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)


json_schema = { # type: ignore[no-untyped-call]
    "title": "StructuredResponse",
    "type": "object",
    "description": "A structured response model for the Google Generative AI response.",
    "properties": {
        "user_input": {
            "type": "string",
            "description": "the user input for response"
        },
        "thinking": {
            "type": "string",
            "description": "think about the problem and provide for the user input"
        },
        "response": {
            "type": "string",
            "description": "the response from the model"
        }
    },
}
setup_structure = llm.with_structured_output(json_schema)  # type: ignore[call-arg]
response = setup_structure.invoke("tell me something about math proofs")  # type: ignore[call-arg]
print(response)  # type: ignore[no-untyped-call]

"""
output:
{'response': 'Mathematical proofs are rigorous arguments that demonstrate the truth of a statement. They rely on logical deductions from axioms and previously proven theorems. Different proof techniques exist, such as direct proof, proof by contradiction, and proof by induction.', 
'thinking': 'I will provide a response about mathematical proofs. '}
"""


#you can see the response is a dictionary with keys 'response' and 'thinking' but this missing the 'user_input' key
# if you want to add the 'user_input' key to the response, you can do it like this:
json_schema = { # type: ignore[no-untyped-call]
    "title": "StructuredResponse",
    "type": "object",
    "description": "A structured response model for the Google Generative AI response.",
    "properties": {
        "user_input": {
            "type": "string",
            "description": "the user input for response"
        },
        "thinking": {
            "type": "string",
            "description": "think about the problem and provide for the user input"
        },
        "response": {
            "type": "string",
            "description": "the response from the model"
        }
    },
    "required": ["user_input", "thinking", "response"]  # Ensure all fields are required
}
setup_structure = llm.with_structured_output(json_schema)  # type: ignore[call-arg]
response = setup_structure.invoke("tell me something about math proofs")  # type: ignore[call-arg]
print(response)  # type: ignore[no-untyped-call]

"""
# Now the response will include 'user_input' as well
# Output:
{'user_input': 'tell me something about math proofs', 
'response': 'Math proofs are logical arguments that demonstrate the truth of a mathematical statement. They start with axioms (basic assumptions) and use deductive reasoning to reach a conclusion. Different types of proofs exist, like direct proofs, proofs by contradiction, and proofs by induction.', 
'thinking': 'The user is asking a general question about math proofs, so I should provide a concise and informative overview of the topic, covering the basic definition, purpose, and some common types of proofs.'}
"""

#another that you can see though the reponse have the 'user_input' key, but the order of the keys is not what you want
# if you want to change the order of the keys, you can do it like this:

ordered_keys = ["user_input", "thinking", "response"]
ordered_response = {key: response[key] for key in ordered_keys}  # type: ignore[no-untyped-call]
print(ordered_response)  # type: ignore[no-untyped-call]

"""so after that you can get the ordered output like this:
Output:
{'user_input': 'tell me something about math proofs',
 'thinking': 'The user is asking a general question about math proofs, so I should provide a concise and informative overview of the topic, covering the basic definition, purpose, and some common types of proofs.',
 'response': 'Math proofs are logical arguments that demonstrate the truth of a mathematical statement. They start with axioms (basic assumptions) and use deductive reasoning to reach a conclusion. Different types of proofs exist, like direct proofs, proofs by contradiction, and proofs by induction.'}
"""