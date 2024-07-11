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
There are multiple Lead Information given below ,you need to generate script based on question asked about particular lead .Verify it using identifying car details with lead information.
You only need to create one script at one time ,just check about which lead is question  then generate script for that only.
Here are all the leads with their information:

1. Lead 1 Information:

    • Prospect Status: Interested
    • Request Date: 2024-06-23T15:49:48-05:00

Vehicles of Interest:

    1. Interest: Buy, Status: Used, Year: 2022, Make: Kia, Model: Telluride, Trim: EX
    2. Interest: Trade-in, Status: New, Odometer Status: Replaced, Odometer Units: mi

Customer Details:

    • First Name: Tyler
    • Last Name: Uebele
    • Phone: 704-443-8469
    • Address: 204 E Franklin, Monroe, NC 28212

Vendor Name: Xanadu Automotive (TESTER)

Provider Details:

    • ID: 1
    • Source: Unknown

Lead Inventory Items:

    1. Inventory ID: 7302674, VIN: 5XYP34HC0NG195109, New/Used: Used, Year: 2022, Make: GMC, Model: Sierra 1500, Transmission: Automatic, Odometer: 13,693 mi, Color: Black, Price: $50,695, Description: 8-Speed Automatic, 4WD, Black Cloth. Features: Preferred Equipment Group 1LT|3.23 Rear Axle Ratio|Auto-Locking Rear Differential|Wheels: 10" x 8" Bright Silver Painted Aluminum|Wheels: 20" x 8.5" Bright Silver Painted Aluminum|Wheels: 22" x 9" Painted Aluminum|40/20/40 Front Split-Bench Seat|Cloth Seat Trim|10-Way Power Driver Seat w/Lumbar|Convenience Package II|All Star Edition Plus|Radio: Chevrolet Infotainment 3 Premium System|Electronic Cruise Control|Electric Rear-Window Defogger|Color-Keyed Carpeting Floor Covering|All-Weather Floor Liner (LPO) (AAK)|120-Volt Interior Power Outlet|Z71 Off-Road & Protection Package|Protection Package|Chevytec Spray-On Black Bedliner|Deep-Tinted Glass|Front License Plate Kit|LED Cargo Area Lighting|EZ Lift Power Lock & Release Tailgate|Rear Wheelhouse Liners|6" Rectangular Chrome Tubular Assist Steps (LPO)|Front Black Bowtie Emblem (LPO)|Standard Suspension Package|High Capacity Suspension Package|Z71 Off-Road Package|Trailering Package|Integrated Trailer Brake Controller|Remote Start Package|Skid Plates|Heavy-Duty Air Filter|SiriusXM w/360L|Power Sliding Rear Window w/Rear Defogger|Rear 60/40 Folding Bench Seat (Folds Up)|Chevrolet Connected Access Capable|Power Front Windows w/Passenger Express Down|Power Rear Windows w/Express Down|Keyless Open & Start|Power Front Windows w/Driver Express Up/Down|Front Rubberized Vinyl Floor Mats|Rear Rubberized-Vinyl Floor Mats|Bluetooth® For Phone|Remote Vehicle Starter System|Dual-Zone Automatic Climate Control|Hitch Guidance|Inside Rear-View Mirror w/Tilt|Heated Power-Adjustable Outside Mirrors|Chrome Mirror Caps|Hill Descent Control|Heated Driver & Front Outboard Passenger Seats|External Engine Oil Cooler|120-Volt Bed Mounted Power Outlet|Heated Steering Wheel|Auxiliary External Transmission Oil Cooler|220 Amp Alternator|170 Amp Alternator|Electrical Steering Column Lock|Dual Exhaust w/Polished Outlets|Wrapped Steering Wheel|Single-Speed Transfer Case|2-Speed Transfer Case|Convenience Package|All-Star Edition|Chevy Safety Assist|Hitch Guidance w/Hitch View|Standard Tailgate|IntelliBeam Automatic High Beam On/Off|Dual Rear USB Ports (Charge Only)|12.3" Multicolor Reconfigurable Digital Display|OnStar & Chevrolet Connected Services Capable|Following Distance Indicator|In-Vehicle Trailering System App|Forward Collision Alert|Universal Home Remote|Lane Keep Assist w/Lane Departure Warning|Automatic Emergency Braking|Steering Wheel Audio Controls|Front Pedestrian Braking|Theft Deterrent System (Unauthorized Entry)|HD Rear Vision Camera|Front Frame-Mounted Black Recovery Hooks|Wi-Fi Hot Spot Capable|Auto High-beam Headlights|AM/FM radio: SiriusXM with 360L|Premium audio system: Chevrolet Infotainment 3 Premium|Standard fuel economy fuel type: gasoline|4-Wheel Disc Brakes|6 Speakers|Air Conditioning|Electronic Stability Control|Tachometer|Voltmeter|ABS brakes|Alloy wheels|Automatic temperature control|Brake assist|Bumpers: chrome|Delay-off headlights|Driver door bin|Driver vanity mirror|Dual front impact airbags|Dual front side impact airbags|Front anti-roll bar|Front dual zone A/C|Front reading lights|Front wheel independent suspension|Fully automatic headlights|Heated door mirrors|Heated front seats|Heated steering wheel|Illuminated entry|Low tire pressure warning|Occupant sensing airbag|Outside temperature display|Overhead airbag|Overhead console|Panic alarm|Passenger door bin|Passenger vanity mirror|Power door mirrors|Power driver seat|Power steering|Power windows|Radio data system|Rear reading lights|Rear step bumper|Rear window defroster|Remote keyless entry|Security system|Speed control|Speed-sensing steering|Split folding rear seat|Steering wheel mounted audio controls|Telescoping steering wheel|Tilt steering wheel|Traction control|Trip computer|Variably intermittent wipers|Compass|Front Center Armrest w/Storage

    2. Inventory ID: 7648318, VIN: 5XYP3DHC5LG079451, New/Used: Used, Year: 2024, Make: Buick, Model: Enclave, Transmission: 8-Speed A/T, Odometer: 77,455 mi, Color: Ebony Black, Price: $25,830, Description: 19/27 City/Highway MPG, Preferred Equipment Group 3LT|3.47 Final Drive Axle Ratio|3.49 Final Drive Axle Ratio|Wheels: 18" Grazen Metallic Aluminum|Wheels: 18" High Gloss Black Painted Aluminum|Black Lug Nut & Wheel Lock Kit (LPO)|Perforated Leather-Appointed Seat Trim|Ride & Handling Suspension|Driver Confidence Package|Sound & Technology Package|Not Equipped w/Rear Park Assist|Radio: Chevrolet Infotainment 3 Plus System|Radio: Chevrolet Infotainment 3 Premium System|Power Panoramic Tilt-Sliding Sunroof|Midnight/Sport Edition|Front & Rear Black Bowties|8-Way Power Driver Seat Adjuster|6-Way Power Front Passenger Seat Adjuster|Power Driver Lumbar Control|Inside Rear-View Auto-Dimming Mirror|Outside Heated Power-Adjustable Body-Color Mirrors|Wireless Charging|Heated Driver & Front Passenger Seats|120-Volt Power Outlet|Adaptive Cruise Control|170 Amp Alternator|155 Amp Alternator|2 USB Data Ports w/SD Card Reader|Rear Power Programmable Liftgate|SiriusXM w/360L|Rear Park Assist w/Audible Warning|Rear Cross Traffic Alert|Universal Home Remote|Enhanced Automatic Emergency Braking|Lane Change Alert w/Side Blind Zone Alert|Bose Premium 8-Speaker Audio System Feature|6-Speaker Audio System Feature|HD Surround Vision|Black Roof-Mounted Side Rails|Variably intermittent wipers|Front beverage holders|Auto-dimming Rear-View mirror|Child-Seat-Sensing Airbag|Compass|Auto High-beam Headlights|AM/FM radio: SiriusXM with 360L|Emergency communication system: OnStar and Chevrolet connected services capable|Premium audio system: Chevrolet Infotainment 3 Plus|Apple CarPlay/Android Auto|4-Wheel Disc Brakes|6 Speakers|Air Conditioning|Electronic Stability Control|Front Bucket Seats|Front Center Armrest|Leather Shift Knob|Power Liftgate|Spoiler|Tachometer|Voltmeter|ABS brakes|Alloy wheels|Auto-dimming door mirrors|Automatic temperature control|Brake assist|Bumpers: body-color|Delay-off headlights|Driver door bin|Driver vanity mirror|Dual front impact airbags|Dual front side impact airbags|Four wheel independent suspension|Front anti-roll bar|Front dual zone A/C|Front reading lights|Fully automatic headlights|Garage door transmitter|Heated door mirrors|Heated front seats|Illuminated entry|Knee airbag|Leather steering wheel|Low tire pressure warning|Occupant sensing airbag|Outside temperature display|Overhead airbag|Overhead console|Panic alarm|Passenger door bin|Passenger vanity mirror|Power door mirrors|Power driver seat|Power passenger seat|Power steering|Power windows|Radio data system|Rear anti-roll bar|Rear reading lights|Rear seat center armrest|Rear window defroster|Rear window wiper|Remote keyless entry|Roof rack: rails only|Security system|Speed control|Speed-sensing steering|Split folding rear seat|Steering wheel mounted audio controls|Telescoping steering wheel|Tilt steering wheel|Traction control|Trip computer|Turn signal indicator mirrors

Returning Customer: False

Lead Type: Sales


---

2. Lead Information 2:

    • Prospect Status: Highly Interested
    • Request Date: 2024-05-23T15:49:48-05:00

Vehicles of Interest:

    1. Interest: Lease, Status: New, Year: 2023, Make: GMC, Model: Acadia, Trim: Denali
    2. Interest: Buy, Status: Used, Year: 2022, Make: Chevrolet, Model: Tahoe, Trim: LTZ

Customer Details:

    • First Name: Jane
    • Last Name: Smith
    • Phone: 555-987-6543
    • Address: 456 Elm St, Anytown, CA 90210

Vendor Name: Xanadu Automotive (TESTER)

Provider Details:

    • ID: 2
    • Source: Referral

Lead Inventory Items:

    1. Inventory ID: A98765, VIN: 1GKKNSLS0LZ123456, New/Used: New, Year: 2024, Make: GMC, Model: Acadia, Transmission: 8-Speed Automatic, Odometer: 0 mi, Color: Summit White, Price: $52,000, Description: Denali Ultimate Package|3.6L V6 Engine|8-Speed Automatic Transmission|AWD|Summit White Exterior|Jet Black Interior|20" Polished Aluminum Wheels|Driver Alert Package II|Dual SkyScape 2-Panel Power Sunroof|Wireless Apple CarPlay/Wireless Android Auto|Premium Bose 8-Speaker Audio System|HD Surround Vision|Advanced Adaptive Cruise Control|Heated and Ventilated Front Seats|Power Liftgate|Navigation System|Remote Start.

    2. Inventory ID: B54321, VIN: 1GNSKCKC6NR123456, New/Used: Used, Year: 2022, Make: Chevrolet, Model: Tahoe, Transmission: 10-Speed Automatic, Odometer: 18,450 mi, Color: Black, Price: $62,500, Description: LTZ Preferred Equipment Group|5.3L V8 Engine|10-Speed Automatic Transmission|4WD|Black Exterior|Jet Black Interior|22" Multi-Spoke Gloss Black Wheels|Driver Alert Package|Bose Centerpoint Surround Sound Audio System|Wireless Apple CarPlay/Wireless Android Auto|HD Rear Vision Camera|Power Sunroof|Heated and Ventilated Front Seats|Third Row Seating|Power Folding Rear Seats|Hands-Free Liftgate|Navigation System|Remote Start|Advanced Trailering System.

Returning Customer: True

Lead Type: Lease/Sales


---

3. Lead Information 3:

    • Prospect Status: [Prospect Status]
    • Request Date: 2024-05-23T15:49:48-05:00

Vehicles of Interest:

    1. Interest: Buy, Status: New, Year: 2022, Make: GMC, Model: Canyon, Trim: EX
    2. Interest: Trade-in, Status: New, Odometer Status: Replaced, Odometer Units: mi

Customer Details:

    • First Name: Savy
    • Last Name: James
    • Phone: 874-443-8469
    • Address: 105 E Franklin, Monroe, NC 28312

Vendor Name: Xanadu Automotive (TESTER)

Provider Details:

    • ID: 1
    • Source: Unknown

Lead Inventory Items:

    1. Inventory ID: 8302644, VIN: 6XYP34HC0NG195109, New/Used: New, Year: 2022, Make: Buick, Model: Enclave, Transmission: Automatic, Odometer: 13,693 mi, Color: Sangria, Price: $34,695, Description: 10-Speed Automatic, 4WD, Black Cloth. Features: Preferred Equipment Group 1LT|3.23 Rear Axle Ratio|Auto-Locking Rear Differential|Wheels: 20" x 8" Bright Silver Painted Aluminum|Wheels: 18" x 8.5" Bright Silver Painted Aluminum|Wheels: 20" x 9" Painted Aluminum|40/20/40 Front Split-Bench Seat|Cloth Seat Trim|10-Way Power Driver Seat w/Lumbar|Convenience Package II|All Star Edition Plus|Radio: Chevrolet Infotainment 3 Premium System|Electronic Cruise Control|Electric Rear-Window Defogger|Color-Keyed Carpeting Floor Covering|All-Weather Floor Liner (LPO) (AAK)|120-Volt Interior Power Outlet|Z71 Off-Road & Protection Package|Protection Package|Chevytec Spray-On Black Bedliner|Deep-Tinted Glass|Front License Plate Kit|LED Cargo Area Lighting|EZ Lift Power Lock & Release Tailgate|Rear Wheelhouse Liners|6" Rectangular Chrome Tubular Assist Steps (LPO)|Front Black Bowtie Emblem (LPO)|Standard Suspension Package|High Capacity Suspension Package|Z71 Off-Road Package|Trailering Package|Integrated Trailer Brake Controller|Remote Start Package|Skid Plates|Heavy-Duty Air Filter|SiriusXM w/360L|Power Sliding Rear Window w/Rear Defogger|Rear 60/40 Folding Bench Seat (Folds Up)|Chevrolet Connected Access Capable|Power Front Windows w/Passenger Express Down|Power Rear Windows w/Express Down|Keyless Open & Start|Power Front Windows w/Driver Express Up/Down|Front Rubberized Vinyl Floor Mats|Rear Rubberized-Vinyl Floor Mats|Bluetooth® For Phone|Remote Vehicle Starter System|Dual-Zone Automatic Climate Control|Hitch Guidance|Inside Rear-View Mirror w/Tilt|Heated Power-Adjustable Outside Mirrors|Chrome Mirror Caps|Hill Descent Control|Heated Driver & Front Outboard Passenger Seats|External Engine Oil Cooler|120-Volt Bed Mounted Power Outlet|Heated Steering Wheel|Auxiliary External Transmission Oil Cooler|220 Amp Alternator|170 Amp Alternator|Electrical Steering Column Lock|Dual Exhaust w/Polished Outlets|Wrapped Steering Wheel|Single-Speed Transfer Case|2-Speed Transfer Case|Convenience Package|All-Star Edition|Chevy Safety Assist|Hitch Guidance w/Hitch View|Standard Tailgate|IntelliBeam Automatic High Beam On/Off|Dual Rear USB Ports (Charge Only)|12.3" Multicolor Reconfigurable Digital Display|OnStar & Chevrolet Connected Services Capable|Following Distance Indicator|In-Vehicle Trailering System App|Forward Collision Alert|Universal Home Remote|Lane Keep Assist w/Lane Departure Warning|Automatic Emergency Braking|Steering Wheel Audio Controls|Front Pedestrian Braking|Theft Deterrent System (Unauthorized Entry)|HD Rear Vision Camera|Front Frame-Mounted Black Recovery Hooks|Wi-Fi Hot Spot Capable|Auto High-beam Headlights|AM/FM radio: SiriusXM with 360L|Premium audio system: Chevrolet Infotainment 3 Premium|Standard fuel economy fuel type: gasoline|4-Wheel Disc Brakes|6 Speakers|Air Conditioning|Electronic Stability Control|Tachometer|Voltmeter|ABS brakes|Alloy wheels|Automatic temperature control|Brake assist|Bumpers: chrome|Delay-off headlights|Driver door bin|Driver vanity mirror|Dual front impact airbags|Dual front side impact airbags|Front anti-roll bar|Front dual zone A/C|Front reading lights|Front wheel independent suspension|Fully automatic headlights|Heated door mirrors|Heated front seats|Heated steering wheel|Illuminated entry|Low tire pressure warning|Occupant sensing airbag|Outside temperature display|Overhead airbag|Overhead console|Panic alarm|Passenger door bin|Passenger vanity mirror|Power door mirrors|Power driver seat|Power steering|Power windows|Radio data system|Rear reading lights|Rear step bumper|Rear window defroster|Remote keyless entry|Security system|Speed control|Speed-sensing steering|Split folding rear seat|Steering wheel mounted audio controls|Telescoping steering wheel|Tilt steering wheel|Traction control|Trip computer|Variably intermittent wipers|Compass|Front Center Armrest w/Storage

    2. Inventory ID: 9648355, VIN: 9XYP3DHC5LG079451, New/Used: Used, Year: 2024, Make: Buick, Model: Enclave, Transmission: 8-Speed A/T, Odometer: 77,455 mi, Color: Ebony Black, Price: $85,930, Description: 19/27 City/Highway MPG, Preferred Equipment Group 3LT|3.47 Final Drive Axle Ratio|3.49 Final Drive Axle Ratio|Wheels: 20" Grazen Metallic Aluminum|Wheels: 20" High Gloss Black Painted Aluminum|Black Lug Nut & Wheel Lock Kit (LPO)|Perforated Leather-Appointed Seat Trim|Ride & Handling Suspension|Driver Confidence Package|Sound & Technology Package|Not Equipped w/Rear Park Assist|Radio: Chevrolet Infotainment 3 Plus System|Radio: Chevrolet Infotainment 3 Premium System|Power Panoramic Tilt-Sliding Sunroof|Midnight/Sport Edition|Front & Rear Black Bowties|8-Way Power Driver Seat Adjuster|6-Way Power Front Passenger Seat Adjuster|Power Driver Lumbar Control|Inside Rear-View Auto-Dimming Mirror|Outside Heated Power-Adjustable Body-Color Mirrors|Wireless Charging|Heated Driver & Front Passenger Seats|120-Volt Power Outlet|Adaptive Cruise Control|170 Amp Alternator|155 Amp Alternator|2 USB Data Ports w/SD Card Reader|Rear Power Programmable Liftgate|SiriusXM w/360L|Rear Park Assist w/Audible Warning|Rear Cross Traffic Alert|Universal Home Remote|Enhanced Automatic Emergency Braking|Lane Change Alert w/Side Blind Zone Alert|Bose Premium 8-Speaker Audio System Feature|6-Speaker Audio System Feature|HD Surround Vision|Black Roof-Mounted Side Rails|Variably intermittent wipers|

Front beverage holders|Auto-dimming Rear-View mirror|Child-Seat-Sensing Airbag|Compass|Auto High-beam Headlights|AM/FM radio: SiriusXM with 360L|Emergency communication system: OnStar and Chevrolet connected services capable|Premium audio system: Chevrolet Infotainment 3 Plus|Apple CarPlay/Android Auto|4-Wheel Disc Brakes|6 Speakers|Air Conditioning|Electronic Stability Control|Front Bucket Seats|Front Center Armrest|Leather Shift Knob|Power Liftgate|Spoiler|Tachometer|Voltmeter|ABS brakes|Alloy wheels|Auto-dimming door mirrors|Automatic temperature control|Brake assist|Bumpers: body-color|Delay-off headlights|Driver door bin|Driver vanity mirror|Dual front impact airbags|Dual front side impact airbags|Four wheel independent suspension|Front anti-roll bar|Front dual zone A/C|Front reading lights|Fully automatic headlights|Garage door transmitter|Heated door mirrors|Heated front seats|Illuminated entry|Knee airbag|Leather steering wheel|Low tire pressure warning|Occupant sensing airbag|Outside temperature display|Overhead airbag|Overhead console|Panic alarm|Passenger door bin|Passenger vanity mirror|Power door mirrors|Power driver seat|Power passenger seat|Power steering|Power windows|Radio data system|Rear anti-roll bar|Rear reading lights|Rear seat center armrest|Rear window defroster|Rear window wiper|Remote keyless entry|Roof rack: rails only|Security system|Speed control|Speed-sensing steering|Split folding rear seat|Steering wheel mounted audio controls|Telescoping steering wheel|Tilt steering wheel|Traction control|Trip computer|Turn signal indicator mirrors

Returning Customer: False

Lead Type: Sales


---

4. Lead Information 4:

    • Prospect Status: Interested
    • Request Date: 2024-07-11T13:00:00-05:00

Vehicles of Interest:

    1. Interest: Buy, Status: New, Year: 2023, Make: Buick, Model: Envision, Trim: Avenir

Customer Details:

    • First Name: John
    • Last Name: Doe
    • Phone: 555-123-4567
    • Address: 123 Main St, Springfield, IL 62701

Vendor Name: Xanadu Automotive (TESTER)

Provider Details:

    • ID: 1
    • Source: Unknown

Lead Inventory Items:

    1. Inventory ID: B27374, VIN: LRBFZSR45PD233486, New/Used: New, Year: 2023, Make: Buick, Model: Envision, Transmission: 9-Speed Automatic, Odometer: 22/29 City/Highway MPG, Color: Moonstone Gray Metallic, Price: $43,605, Description: Preferred Equipment Group 1SU|3.47 Final Drive Axle Ratio|Wheels: 20" Aluminum w/Avenir Pearl Nickel Finish|Front Bucket Seats|Perforated Leather-Appointed Seat Trim|Radio: Buick Infotainment System AM/FM w/Nav|8-Way Power Driver Seat Adjuster|Front Passenger 8-Way Power Seat Adjuster|Heated Driver & Front Passenger Seats|Heated Rear Outboard Seating Positions|Ventilated Driver & Front Passenger Seats|Memory Card Receptacle Audio System Feature|Front Bin Center Console USB Ports|Wireless Apple CarPlay/Wireless Android Auto|IntelliBeam Headlamp Control w/Auto High Beam|SiriusXM Radio|HD Radio|Following Distance Indicator|Forward Collision Alert|Lane Keep Assist w/Lane Departure Warning|Front Pedestrian Braking|Bose Premium 9-Speaker Audio System Feature|USB Charging-Only Ports|4-Wheel Disc Brakes|9 Speakers|Air Conditioning|Electronic Stability Control|Front Center Armrest|Navigation System|Power Liftgate|Spoiler|ABS brakes|Adjustable head restraints: driver and passenger w/tilt|Alloy wheels|Auto-dimming door mirrors|Automatic temperature control|Brake assist|Bumpers: body-color|Delay-off headlights|Driver door bin|Driver vanity mirror|Dual front impact airbags|Dual front side impact airbags|Four wheel independent suspension|Front anti-roll bar|Front dual zone A/C|Front reading lights|Fully automatic headlights|Garage door transmitter|Heated door mirrors|Heated front seats|Heated rear seats|Heated steering wheel|Illuminated entry|Knee airbag|Low tire pressure warning|Memory seat|Occupant sensing airbag|Outside temperature display|Overhead airbag|Overhead console|Panic alarm|Passenger door bin|Passenger vanity mirror|Power door mirrors|Power driver seat|Power moonroof|Power passenger seat|Power steering|Power windows|Radio data system|Rear anti-roll bar|Rear reading lights|Rear seat center armrest|Rear window defroster|Rear window wiper|Remote keyless entry|Roof rack: rails only|Security system|Speed control|Split folding rear seat|Steering wheel mounted audio controls|Telescoping steering wheel|Tilt steering wheel|Traction control|Trip computer|Variably intermittent wipers|Ventilated front seats|Auto-dimming Rear-View mirror|Heads-Up Display|Compass|AM/FM radio: SiriusXM|Premium audio system: Bose|Exterior Parking Camera Rear|Auto High-beam Headlights|Emergency communication system: OnStar and Buick connected services capable|Apple CarPlay/Android Auto.

Returning Customer: False

Lead Type: Sales


---

5. Lead Information 5:

    • Prospect Status: Highly Interested
    • Request Date: 2024-07-11T14:30:00-05:00

Vehicles of Interest:

    1. Interest: Lease, Status: New, Year: 2024, Make: GMC, Model: Acadia, Trim: Denali
    2. Interest: Buy, Status: Used, Year: 2022, Make: Chevrolet, Model: Tahoe, Trim: LTZ

Customer Details:

    • First Name: Jane
    • Last Name: Smith
    • Phone: 555-987-6543
    • Address: 456 Elm St, Anytown, CA 90210

Vendor Name: Xanadu Automotive (TESTER)

Provider Details:

    • ID: 2
    • Source: Referral

Lead Inventory Items:

    1. Inventory ID: A98765, VIN: 1GKKNSLS0LZ123456, New/Used: New, Year: 2024, Make: GMC, Model: Acadia, Transmission: 8-Speed Automatic, Odometer: 0 mi, Color: Summit White, Price: $52,000, Description: Denali Ultimate Package|3.6L V6 Engine|8-Speed Automatic Transmission|AWD|Summit White Exterior|Jet Black Interior|20" Polished Aluminum Wheels|Driver Alert Package II|Dual SkyScape 2-Panel Power Sunroof|Wireless Apple CarPlay/Wireless Android Auto|Premium Bose 8-Speaker Audio System|HD Surround Vision|Advanced Adaptive Cruise Control|Heated and Ventilated Front Seats|Power Liftgate|Navigation System|Remote Start.

    2. Inventory ID: B54321, VIN: 1GNSKCKC6NR123456, New/Used: Used, Year: 2022, Make: Chevrolet, Model: Tahoe, Transmission: 10-Speed Automatic, Odometer: 18,450 mi, Color: Black, Price: $62,500, Description: LTZ Preferred Equipment Group|5.3L V8 Engine|10-Speed Automatic Transmission|4WD|Black Exterior|Jet Black Interior|22" Multi-Spoke Gloss Black Wheels|Driver Alert Package|Bose Centerpoint Surround Sound Audio System|Wireless Apple CarPlay/Wireless Android Auto|HD Rear Vision Camera|Power Sunroof|Heated and Ventilated Front Seats|Third Row Seating|Power Folding Rear Seats|Hands-Free Liftgate|Navigation System|Remote Start|Advanced Trailering System.

Returning Customer:** True

Lead Type:** Lease/Sales

---
Comments: {comments}

Instructions:
Create a script with the following sections:

    - Greeting
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
