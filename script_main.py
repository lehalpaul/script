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
    st.title("Script Assistant")

   
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
        
def run_conversation(comments):
    llm = ChatOpenAI(temperature=0.9, model="gpt-4", streaming=True)


    template="""
You are an AI assistant tasked with creating phone scripts for sales representatives at Xanadu Automotive, focusing on engaging potential customers based on their expressed interests and provided comments. Your scripts will help manage inbound leads by responding accurately and efficiently.

Task:
Generate a phone script for an inbound lead. Use the following information to craft a friendly and professional script that includes necessary sections like greeting, vehicle information, and closing statements.

  "source": "Manifest List",
  "message_type": "Service",
  "dealer_label": "!!111 Xanadu & Automotive (TESTER)",
  "dealer_address": "204 E Franklin, Monroe, NC 28212",
  "dealer_phone": "317-555-1212",
  "dealer_contact_name": "Guilian MINION",
  "dealer_contact_phone": "317-555-1212",
  "dealer_url": "",
  "target_vehicle_new_used": "used",
  "target_vehicle_year": "2016",
  "target_vehicle_make": "Buick",
  "target_vehicle_model": "Encore",
  "target_vehicle_vin": "",
  "target_vehicle_trim": "",
  "trade_vehicle_year": "2000",
  "trade_vehicle_make": "Acura",
  "trade_vehicle_model": "",
  "trade_vehicle_vin": "",
  "inventory": "",
  
  "returning_customer": "1",
  "trade_vehicle_new_used": "",
  "trade_vehicle_trim": ""



Comments: {comments}

Instructions:
Create a script with the following sections:

    - Greet the lead with their name related to the lead data .
    - Introduction (Incorporate vehicles of interest and relevant customer comments)
    - Returning Customer Acknowledgement (if applicable)
    - Questions to Understand Lead’s Needs
    - Closing Statement (Include scheduling for a follow-up appointment or test drive)
Ensure  script is formatted with clear labels for each section.
"""


    prompt_temp = ChatPromptTemplate.from_template(template)

    memory = ConversationBufferMemory(memory_key= "chat_history",return_messages=True)
    conversation = LLMChain(llm=llm,prompt=prompt_temp,verbose=False,memory=memory)


    ai_response =conversation.run(comments)

    
    

    return ai_response

if __name__ == "__main__":
    main()
