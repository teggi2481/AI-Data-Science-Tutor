from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import time

# Load environment variables
load_dotenv()

# Create LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_output_tokens=500
)

#################################################################
################ Testing the Gemini Model
#################################################################
# Send a message to Gemini
response = llm.invoke("Hello, how are you?")

# Display Gemini's response
print("Gemini's Response:")
print(response.content)
#################################################################

messages = [
    SystemMessage(
        content="""
        You are an expert Data Science tutor.
        Explain concepts in simple language and ensure to generate the crisp answers within 100 words and not more than that.
        """
    )
]

print("\nAI Data Science Tutor")
print("Type 'exit' to quit\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    messages.append(
        HumanMessage(content=user_input)
    )

    try:
        print("\nWaiting for LLM response...")

        start_time = time.time()

        response = llm.invoke(messages)

        elapsed_time = round(time.time() - start_time, 2)

        print(f"\nResponse received in {elapsed_time} seconds")
        print("\nTutor:")
        print(response.content)

        messages.append(
            AIMessage(content=response.content)
        )

    except TimeoutError:
        print("\nERROR: Request timed out.")

    except ConnectionError:
        print("\nERROR: Unable to connect to Gemini API.")

    except Exception as ex:
        print(f"\nUnexpected Error: {str(ex)}")