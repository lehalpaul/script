import streamlit as st
import os
import openai
import sys
sys.path.append('/path/to/langchain')

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from langchain.document_loaders import CSVLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def main():
    
    openai.api_key=os.environ["OPENAI_API_KEY"]
    st.title("Auto Dealership AI Assistant")

   
    if "messages" not in st.session_state:
        st.session_state.messages = []

   
    with st.form("user_input_form"):
        user_input = st.text_input("How can I assist you today?", key="input")
        submit_button = st.form_submit_button("Submit")

    if submit_button and user_input:
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = run_conversation(user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

   
    for message in reversed(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
def run_conversation(question):
    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-1106", streaming=True)
    loader = CSVLoader(file_path="cronicchevrolet.csv")
    data = loader.load()
    vectordb = FAISS.from_documents(data, OpenAIEmbeddings())
    retriever = vectordb.as_retriever()

    template = """
    
You are an AI assistant for a sales representative at an auto car dealership, who is going to generate a script that a sales person will read while calling the lead. Your primary role is to manage inbound leads by engaging with potential customers.
 Your task is to understand and respond to user comments accurately and efficiently.
      I need your help to create a phone script for an inbound lead at our auto dealership. Here is the information about the lead:

Lead Information:

	•	Prospect Status: [Prospect Status]
	•	Request Date: 2024-05-23T15:49:48-05:00

Vehicles of Interest:

	1.	Interest: Buy
	•	Status: Used
	•	Year: 2022
	•	Make: Kia
	•	Model: Telluride
	•	Trim: EX
	2.	Interest: Trade-in
	•	Status: New
	•	Odometer Status: Replaced
	•	Odometer Units: mi

Customer Details:

	•	First Name: Tyler
	•	Last Name: Uebele
	•	Phone: 704-443-8469
	•	Address: 204 E Franklin, Monroe, NC 28212

Vendor Name: !!111 Xanadu Automotive (TESTER)

Provider Details:

	•	ID: 1
	•	Source: Unknown

Lead Inventory Items:

	1.	Inventory ID: 7302674
	•	Dealer ID: 412294
	•	Stock Number: G240307A
	•	VIN: 5XYP34HC0NG195109
	•	New/Used: Used
	•	Year: 2022
	•	Make: Kia
	•	Model: Telluride
	•	Transmission: Automatic
	•	Odometer: 13693
	•	Color: Sangria
	•	Price: $34,695
	•	URL: [Inventory Item 1 URL]
	2.	Inventory ID: 7648318
	•	Dealer ID: 412294
	•	Stock Number: D241033B
	•	VIN: 5XYP3DHC5LG079451
	•	New/Used: Used
	•	Year: 2020
	•	Make: Kia
	•	Model: Telluride
	•	Transmission: 8-Speed A/T
	•	Odometer: 77455
	•	Color: Ebony Black
	•	Price: $25,830
	•	URL: [Inventory Item 2 URL]

Returning Customer: False
Lead Type: Sales

Comments: {comments}

Please create a friendly and professional phone script that includes a greeting, questions to understand the lead’s needs better, information about the vehicles of interest, and a closing statement to schedule a follow-up appointment or test drive.

Context:
The Phone script will also include inventory information .Please understand the comments first then generate script according to that.
Do not role play the script, give me one script so that I can guide the rep in my dealership.
You have to provide formatted script by labelling such as Greeting ,Introduction,Closing Statement, Returning Customer Acknowledgement ,Next Steps etc.

    
    
    
    """
    prompt_temp = ChatPromptTemplate.from_template(template)

    memory = ConversationBufferMemory(memory_key= "chat_history",return_messages=True)
    conversation = LLMChain(llm=llm,prompt=prompt_temp,verbose=False,memory=memory)


    comments = "Hi"
    ai_response =conversation.run(comments)

    
    

    return ai_response

if __name__ == "__main__":
    main()