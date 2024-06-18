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
        
def run_conversation(comments):
    llm = ChatOpenAI(temperature=0.9, model="gpt-4", streaming=True)


    template = """
    
You are an AI assistant for a sales representative at an auto car dealership, who is going to generate a script that a sales person will read while calling the lead. Your primary role is to manage inbound leads by engaging with potential customers.
 Your task is to understand and respond to user comments accurately and efficiently.
      I need your help to create a phone script for an inbound lead at our auto dealership. Here is the information about the lead:

  
Lead Information:
    • Prospect Status: [Prospect Status]
    • Request Date: 2024-05-23T15:49:48-05:00

Vehicles of Interest:
    1. Interest: Buy, Status: Used, Year: 2022, Make: Kia, Model: Telluride, Trim: EX
    2. Interest: Trade-in, Status: New, Odometer Status: Replaced, Odometer Units: mi

Customer Details:
    •	First Name: Tyler
•	Last Name: Uebele
•	Phone: 704-443-8469
•	Address: 204 E Franklin, Monroe, NC 28212

Vendor Name: Xanadu Automotive (TESTER)

Provider Details:
    • ID: 1
    • Source: Unknown

Lead Inventory Items:
    1. Inventory ID: 7302674, VIN: 5XYP34HC0NG195109, New/Used: Used, Year: 2022, Make: Kia,
       Model: Telluride, Transmission: Automatic, Odometer: 13,693 mi, Color: Sangria, Price: $34,695, Description:10-Speed Automatic, 4WD, Black Cloth.  
	Features:Preferred Equipment Group 1LT|3.23 Rear Axle Ratio|Auto-Locking Rear Differential|Wheels: 17" x 8" Bright Silver Painted Aluminum|Wheels: 18" x 8.5" Bright Silver Painted Aluminum|Wheels: 20" x 9" Painted Aluminum|40/20/40 Front Split-Bench Seat|Cloth Seat Trim|10-Way Power Driver Seat w/Lumbar|Convenience Package II|All Star Edition Plus|Radio: Chevrolet Infotainment 3 Premium System|Electronic Cruise Control|Electric Rear-Window Defogger|Color-Keyed Carpeting Floor Covering|All-Weather Floor Liner (LPO) (AAK)|120-Volt Interior Power Outlet|Z71 Off-Road & Protection Package|Protection Package|Chevytec Spray-On Black Bedliner|Deep-Tinted Glass|Front License Plate Kit|LED Cargo Area Lighting|EZ Lift Power Lock & Release Tailgate|Rear Wheelhouse Liners|6" Rectangular Chrome Tubular Assist Steps (LPO)|Front Black Bowtie Emblem (LPO)|Standard Suspension Package|High Capacity Suspension Package|Z71 Off-Road Package|Trailering Package|Integrated Trailer Brake Controller|Remote Start Package|Skid Plates|Heavy-Duty Air Filter|SiriusXM w/360L|Power Sliding Rear Window w/Rear Defogger|Rear 60/40 Folding Bench Seat (Folds Up)|Chevrolet Connected Access Capable|Power Front Windows w/Passenger Express Down|Power Rear Windows w/Express Down|Keyless Open & Start|Power Front Windows w/Driver Express Up/Down|Front Rubberized Vinyl Floor Mats|Rear Rubberized-Vinyl Floor Mats|Bluetooth¬Æ For Phone|Remote Vehicle Starter System|Dual-Zone Automatic Climate Control|Hitch Guidance|Inside Rear-View Mirror w/Tilt|Heated Power-Adjustable Outside Mirrors|Chrome Mirror Caps|Hill Descent Control|Heated Driver & Front Outboard Passenger Seats|External Engine Oil Cooler|120-Volt Bed Mounted Power Outlet|Heated Steering Wheel|Auxiliary External Transmission Oil Cooler|220 Amp Alternator|170 Amp Alternator|Electrical Steering Column Lock|Dual Exhaust w/Polished Outlets|Wrapped Steering Wheel|Single-Speed Transfer Case|2-Speed Transfer Case|Convenience Package|All-Star Edition|Chevy Safety Assist|Hitch Guidance w/Hitch View|Standard Tailgate|IntelliBeam Automatic High Beam On/Off|Dual Rear USB Ports (Charge Only)|12.3" Multicolor Reconfigurable Digital Display|OnStar & Chevrolet Connected Services Capable|Following Distance Indicator|In-Vehicle Trailering System App|Forward Collision Alert|Universal Home Remote|Lane Keep Assist w/Lane Departure Warning|Automatic Emergency Braking|Steering Wheel Audio Controls|Front Pedestrian Braking|Theft Deterrent System (Unauthorized Entry)|HD Rear Vision Camera|Front Frame-Mounted Black Recovery Hooks|Wi-Fi Hot Spot Capable|Auto High-beam Headlights|AM/FM radio: SiriusXM with 360L|Premium audio system: Chevrolet Infotainment 3 Premium|Standard fuel economy fuel type: gasoline|4-Wheel Disc Brakes|6 Speakers|Air Conditioning|Electronic Stability Control|Tachometer|Voltmeter|ABS brakes|Alloy wheels|Automatic temperature control|Brake assist|Bumpers: chrome|Delay-off headlights|Driver door bin|Driver vanity mirror|Dual front impact airbags|Dual front side impact airbags|Front anti-roll bar|Front dual zone A/C|Front reading lights|Front wheel independent suspension|Fully automatic headlights|Heated door mirrors|Heated front seats|Heated steering wheel|Illuminated entry|Low tire pressure warning|Occupant sensing airbag|Outside temperature display|Overhead airbag|Overhead console|Panic alarm|Passenger door bin|Passenger vanity mirror|Power door mirrors|Power driver seat|Power steering|Power windows|Radio data system|Rear reading lights|Rear step bumper|Rear window defroster|Remote keyless entry|Security system|Speed control|Speed-sensing steering|Split folding rear seat|Steering wheel mounted audio controls|Telescoping steering wheel|Tilt steering wheel|
	Traction control|Trip computer|Variably intermittent wipers|Compass|Front Center Armrest w/Storage
 
    2. Inventory ID: 7648318, VIN: 5XYP3DHC5LG079451, New/Used: Used, Year: 2020, Make: Kia,
       Model: Telluride, Transmission: 8-Speed A/T, Odometer: 77,455 mi, Color: Ebony Black, Price: $25,830, 
       Description:19/27 City/Highway MPG   ,Description:Preferred Equipment Group 3LT|3.47 Final Drive Axle Ratio|3.49 Final Drive Axle Ratio|Wheels: 18" Grazen Metallic Aluminum|Wheels: 18" High Gloss Black Painted Aluminum|Black Lug Nut & Wheel Lock Kit (LPO)|Perforated Leather-Appointed Seat Trim|Ride & Handling Suspension|Driver Confidence Package|Sound & Technology Package|Not Equipped w/Rear Park Assist|Radio: Chevrolet Infotainment 3 Plus System|Radio: Chevrolet Infotainment 3 Premium System|Power Panoramic Tilt-Sliding Sunroof|Midnight/Sport Edition|Front & Rear Black Bowties|8-Way Power Driver Seat Adjuster|6-Way Power Front Passenger Seat Adjuster|Power Driver Lumbar Control|Inside Rear-View Auto-Dimming Mirror|Outside Heated Power-Adjustable Body-Color Mirrors|Wireless Charging|Heated Driver & Front Passenger Seats|120-Volt Power Outlet|Adaptive Cruise Control|170 Amp Alternator|155 Amp Alternator|2 USB Data Ports w/SD Card Reader|Rear Power Programmable Liftgate|SiriusXM w/360L|Rear Park Assist w/Audible Warning|Rear Cross Traffic Alert|Universal Home Remote|Enhanced Automatic Emergency Braking|Lane Change Alert w/Side Blind Zone Alert|Bose Premium 8-Speaker Audio System Feature|6-Speaker Audio System Feature|HD Surround Vision|Black Roof-Mounted Side Rails|Variably intermittent wipers|Front beverage holders|Auto-dimming Rear-View mirror|Child-Seat-Sensing Airbag|Compass|Auto High-beam Headlights|AM/FM radio: SiriusXM with 360L|Emergency communication system: OnStar and Chevrolet connected services capable|Premium audio system: Chevrolet Infotainment 3 Plus|Apple CarPlay/Android Auto|4-Wheel Disc Brakes|6 Speakers|Air Conditioning|Electronic Stability Control|Front Bucket Seats|Front Center Armrest|Leather Shift Knob|Power Liftgate|Spoiler|Tachometer|Voltmeter|ABS brakes|Alloy wheels|Auto-dimming door mirrors|Automatic temperature control|Brake assist|Bumpers: body-color|Delay-off headlights|Driver door bin|Driver vanity mirror|Dual front impact airbags|Dual front side impact airbags|Four wheel independent suspension|Front anti-roll bar|Front dual zone A/C|Front reading lights|Fully automatic headlights|Garage door transmitter|Heated door mirrors|Heated front seats|Illuminated entry|Knee airbag|Leather steering wheel|Low tire pressure warning|Occupant sensing airbag|Outside temperature display|Overhead airbag|Overhead console|Panic alarm|Passenger door bin|Passenger vanity mirror|Power door mirrors|Power driver seat|Power passenger seat|Power steering|Power windows|Radio data system|Rear anti-roll bar|Rear reading lights|Rear seat center armrest|Rear window defroster|Rear window wiper|Remote keyless entry|Roof rack: rails only|Security system|Speed control|Speed-sensing steering|Split folding rear seat|Steering wheel mounted audio controls|Telescoping steering wheel|Tilt steering wheel|Traction control|Trip computer|Turn signal indicator mirrors

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


    ai_response =conversation.run(comments)

    
    

    return ai_response

if __name__ == "__main__":
    main()
