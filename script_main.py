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
 Lead Information is  given below ,you need to generate script based on question asked about particular lead.The Phone given in data is Customer Phone.
Lead Information:
Prospect Status: Interested
Vehicles of Interest:
Interest: Buy, Status: Used, Year: 2010, Make: Audi, Model: A5 ,VIN: ,Trim:
Interest: Trade-in, Status: New, Year: 2021, Make: AM General ,VIN: ,Trim:
Customer Details:
First Name: Guilian
Last Name: MINION
Phone: 317-555-1212
Address: 204 E Franklin, Monroe, NC 28212
Vendor Name:
Xanadu Automotive (TESTER)
Provider Details:
ID: 1
Source: Unknown
Lead Inventory Items:
Inventory ID: 8170524, VIN: WBA3B5C53EF959017, New/Used: Used, Year: 2014, Make: BMW, Model: 3 Series, Trim: 328i xDrive, Transmission: Automatic, Odometer: 93118, Color: Jet Black, Price: $10,995
Features: Turbocharged, All Wheel Drive, Power Steering, ABS, 4-Wheel Disc Brakes, Brake Assist, Aluminum Wheels, Tires - Front Performance, Tires - Rear Performance, Heated Mirrors, Power Mirror(s), Integrated Turn Signal Mirrors, Power Folding Mirrors, Rear Defrost, Intermittent Wipers, Variable Speed Intermittent Wipers, Rain Sensing Wipers, Power Door Locks, Daytime Running Lights, Automatic Headlights, Fog Lamps, AM/FM Stereo, CD Player, MP3 Player, HD Radio, Steering Wheel Audio Controls, Auxiliary Audio Input, Power Driver Seat, Power Passenger Seat, Bucket Seats, Pass-Through Rear Seat, Rear Bench Seat, Adjustable Steering Wheel, Trip Computer, Power Windows, Leather Steering Wheel, Keyless Start, Keyless Entry, Universal Garage Door Opener, Cruise Control, Climate Control, Multi-Zone A/C, A/C, Rear A/C, Woodgrain Interior Trim, Premium Synthetic Seats, Auto-Dimming Rearview Mirror, Driver Vanity Mirror, Passenger Vanity Mirror, Driver Illuminated Vanity Mirror, Passenger Illuminated Visor Mirror, Floor Mats, Mirror Memory, Seat Memory, Bluetooth Connection, Immobilizer, Traction Control, Stability Control, Front Side Air Bag, Telematics, Requires Subscription, Tire Pressure Monitor, Driver Air Bag, Passenger Air Bag, Front Head Air Bag, Rear Head Air Bag, Passenger Air Bag Sensor, Knee Air Bag, Child Safety Locks
Photos: https://content.homenetiol.com/2001732/2121940/640x480/f6b615c8fa0649c294f8b835a8e3091b.jpg), Link 2, Link 3
Inventory ID: 8308209, VIN: WBAVB73547VH22567, New/Used: Used, Year: 2007, Make: BMW, Model: 3 Series, Trim: 335i, Transmission: 6-Speed, Odometer: 191019
Features: Turbocharged, Traction Control, Stability Control, Brake Assist, Rear Wheel Drive, Tires - Front Performance, Tires - Rear Performance, Aluminum Wheels, Power Steering, 4-Wheel Disc Brakes, ABS, Sun/Moonroof, Generic Sun/Moonroof, HID headlights, Automatic Headlights, Headlights-Auto-Leveling, Fog Lamps, Daytime Running Lights, Intermittent Wipers, Variable Speed Intermittent Wipers, Rain Sensing Wipers, Premium Synthetic Seats, Power Driver Seat, Power Passenger Seat, Bucket Seats, Seat Memory, Adjustable Steering Wheel, Steering Wheel Audio Controls, Leather Steering Wheel, Tire Pressure Monitor, Immobilizer, Power Windows, Power Door Locks, Cruise Control, Remote Trunk Release, Climate Control, Multi-Zone A/C, A/C, Rear A/C, Rear Defrost, AM/FM Stereo, CD Player, Premium Sound System, MP3 Player, Woodgrain Interior Trim, Driver Vanity Mirror, Passenger Vanity Mirror, Driver Illuminated Vanity Mirror, Passenger Illuminated Visor Mirror, Front Reading Lamps, Rear Reading Lamps, Driver Air Bag, Passenger Air Bag, Passenger Air Bag Sensor, Front Side Air Bag, Front Head Air Bag, Rear Head Air Bag
Photos: https://content.homenetiol.com/2001732/2121940/640x480/3d567a3d92244d78897503c442469f4d.jpg), Link 2
Inventory ID: 8263802, VIN: KL4AMDSLXSB008616, New/Used: New, Year: 2025, Make: Buick, Model: Encore GX, Trim: Sport Touring, Transmission: Variable, Odometer: 3, Color: Moonstone Gray Metallic, Price: $31,080
Features: Adaptive Cruise Control, Wireless Charging, Heated Driver and Front Passenger Seats, Armrest - Rear Center, Audio System - 11" Diagonal HD Color Touchscreen, AM/FM Stereo, Bluetooth Audio Streaming, Voice Command Pass-Through, Wireless Apple CarPlay, Wireless Android Auto, Rear Park Assist, Seat Adjuster - 2-Way Power Driver Lumbar Control, Driver 8-Way Power Seat Adjuster, Transmission - Continuously Variable, Engine - ECOTEC 1.3L Turbo, Moonstone Gray Metallic, Advanced Technology Package, Comfort Package, Front Passenger Flat-Folding Seatback, Ebony Seats and Interior with Santorini Blue Stitching, HD Surround Vision, Sport Touring Preferred Equipment Group, Steering Wheel - Heated, Wheels - 18" Gloss Black Aluminum, Lane Departure Warning, Lane Keeping Assist, Front Collision Mitigation, Front Collision Warning, Automatic Highbeams, Keyless Start, Front Wheel Drive, Power Steering, ABS, 4-Wheel Disc Brakes, Aluminum Wheels, Tires - Front Performance, Tires - Rear Performance, Temporary Spare Tire, Automatic Headlights, Heated Mirrors, Power Mirrors, Privacy Glass, Intermittent Wipers, AM/FM Stereo, Bluetooth Connection, Smart Device Integration, Satellite Radio, WiFi Hotspot, Bucket Seats, Premium Synthetic Seats, Pass-Through Rear Seat, Rear Bench Seat, Floor Mats, Adjustable Steering Wheel, Cruise Control, Steering Wheel Audio Controls, Power Windows, Power Door Locks, Keyless Entry, Security System, MP3 Player, Auxiliary Audio Input, A/C, Rear A/C, Rear Defrost, Driver Vanity Mirror, Passenger Vanity Mirror, Driver Illuminated Vanity Mirror, Passenger Illuminated Visor Mirror, Cargo Shade, Traction Control, Stability Control, Daytime Running Lights, Driver Air Bag, Passenger Air Bag, Front Side Air Bag, Rear Side Air Bag, Front Head Air Bag, Rear Head Air Bag, Knee Air Bag, Passenger Air Bag Sensor, Telematics, Back-Up Camera, Blind Spot Monitor, Cross-Traffic Alert, Child Safety Locks, Driver Restriction Features, Tire Pressure Monitor
Photos: https://content.homenetiol.com/2001732/2121940/640x480/5a13d15ee0c74270b56ad1544eeff092.jpg
Inventory ID: 6240139
VIN: 5GAEVBKW6RJ106787
New/Used: New
Year: 2024
Make: Buick
Model: Enclave
Trim: Premium
Transmission: Automatic
Odometer: 4
Color: White Frost Tricoat
Price: $0 (price not specified)
Features:
Axle - 3.49 Final Drive Ratio
Engine - 3.6L V6, SIDI, VVT Stop/Start
Tires - P255/55R20 All-Season, Blackwall
Whisper Beige Seats with Ebony Interior Accents
Hitch Guidance
Moonroof - Front Power Sliding, transparent glass with rear fixed skylight
Cooling System - Heavy-Duty
White Frost Tricoat
Wheels - 20" Polished Aluminum
Seats - Front Buckets
Transmission - 9-Speed Automatic
Audio System - 8" Diagonal Buick Infotainment System with Navigation
Experience Buick Package
Trailering Package - 5000 lbs.
Premium Preferred Equipment Group
Hitch Guidance with Hitch View
Mirror Memory
Seat Memory
Rear Parking Aid
Blind Spot Monitor
Lane Departure Warning
Cross-Traffic Alert
Lane Keeping Assist
Front Collision Mitigation
Front Collision Warning
Automatic Highbeams
Keyless Start
All Wheel Drive
Power Steering
ABS
4-Wheel Disc Brakes
Aluminum Wheels
Tires - Front All-Season
Tires - Rear All-Season
Temporary Spare Tire
Rear Spoiler
Heated Mirrors
Power Mirror(s)
Auto-Dimming Rearview Mirror
Integrated Turn Signal Mirrors
Power Folding Mirrors
Automatic Headlights
Privacy Glass
Intermittent Wipers
Remote Trunk Release
Power Liftgate
Hands-Free Liftgate
AM/FM Stereo
Navigation System
Satellite Radio
MP3 Player
Bluetooth Connection
Auxiliary Audio Input
Smart Device Integration
Requires Subscription
Premium Sound System
WiFi Hotspot
Bucket Seats
3rd Row Seat
Rear Bucket Seats
Leather Seats
Heated Front Seat(s)
Cooled Front Seat(s)
Heated Rear Seat(s)
Power Driver Seat
Power Passenger Seat
Driver Adjustable Lumbar
Seat-Massage
Passenger Adjustable Lumbar
Floor Mats
Adjustable Steering Wheel
Leather Steering Wheel
Heated Steering Wheel
Cruise Control
Steering Wheel Audio Controls
Power Windows
Heads-Up Display
Power Door Locks
Keyless Entry
Universal Garage Door Opener
Adaptive Cruise Control
Remote Engine Start
Security System
Immobilizer
Climate Control
Multi-Zone A/C
A/C
Rear A/C
Rear Defrost
Power Outlet
Driver Vanity Mirror
Passenger Vanity Mirror
Driver Illuminated Vanity Mirror
Passenger Illuminated Visor Mirror
Traction Control
Stability Control
Daytime Running Lights
Driver Air Bag
Passenger Air Bag
Front Side Air Bag
Front Head Air Bag
Rear Head Air Bag
Passenger Air Bag Sensor
Telematics
Back-Up Camera
Aerial View Display System
Child Safety Locks
Driver Restriction Features
Tire Pressure Monitor
Photos:
Photo 1
Photo 2
Photo 3
Inventory ID: 7057475
VIN: 5GAEVCKW7RJ127914
New/Used: New
Year: 2024
Make: Buick
Model: Enclave
Trim: Avenir
Transmission: Automatic
Odometer: 5
Color: White Frost Tricoat
Price: $57,371
Features:
Seats - Front Buckets
White Frost Tricoat
Wheels - 20" with Pearl Nickel Finish
Audio System - 8" Diagonal Buick Infotainment System with Navigation
Trailering Package - 5000 lbs.
Cooling System - Heavy-Duty
Transmission - 9-Speed Automatic
Hitch Guidance with Hitch View
Axle - 3.49 Final Drive Ratio
Avenir Preferred Equipment Group
Engine - 3.6L V6, SIDI, VVT Stop/Start
Ebony with Ebony Interior Accents, Perforated Leather-Appointed Seats
Hitch Guidance
Mirror Memory
Seat Memory
Rear Parking Aid
Blind Spot Monitor
Lane Departure Warning
Cross-Traffic Alert
Lane Keeping Assist
Front Collision Mitigation
Front Collision Warning
Automatic Highbeams
Keyless Start
All Wheel Drive
Power Steering
ABS
4-Wheel Disc Brakes
Aluminum Wheels
Tires - Front Performance
Tires - Rear Performance
Temporary Spare Tire
Sun/Moonroof
Generic Sun/Moonroof
Rear Spoiler
Heated Mirrors
Power Mirror(s)
Auto-Dimming Rearview Mirror
Integrated Turn Signal Mirrors
Power Folding Mirrors
Automatic Headlights
Privacy Glass
Intermittent Wipers
Variable Speed Intermittent Wipers
Rain Sensing Wipers
Remote Trunk Release
Power Liftgate
Hands-Free Liftgate
AM/FM Stereo
Navigation System
Satellite Radio
MP3 Player
Bluetooth Connection
Auxiliary Audio Input
Smart Device Integration
Requires Subscription
Premium Sound System
WiFi Hotspot
Bucket Seats
3rd Row Seat
Rear Bucket Seats
Leather Seats
Heated Front Seat(s)
Cooled Front Seat(s)
Heated Rear Seat(s)
Power Driver Seat
Power Passenger Seat
Driver Adjustable Lumbar
Seat-Massage
Passenger Adjustable Lumbar
Floor Mats
Adjustable Steering Wheel
Leather Steering Wheel
Heated Steering Wheel
Cruise Control
Steering Wheel Audio Controls
Power Windows
Heads-Up Display
Power Door Locks
Keyless Entry
Universal Garage Door Opener
Adaptive Cruise Control
Remote Engine Start
Security System
Immobilizer
Climate Control
Multi-Zone A/C
A/C
Rear A/C
Rear Defrost
Power Outlet
Driver Vanity Mirror
Passenger Vanity Mirror
Driver Illuminated Vanity Mirror
Passenger Illuminated Visor Mirror
Traction Control
Stability Control
Daytime Running Lights
Driver Air Bag
Passenger Air Bag
Front Side Air Bag
Front Head Air Bag
Rear Head Air Bag
Passenger Air Bag Sensor
Telematics
Back-Up Camera
Aerial View Display System
Child Safety Locks
Driver Restriction Features
Tire Pressure Monitor
Photos:
https://content.homenetiol.com/2001732/2121940/640x480/61b92279fe1044eaa26283fde3fe38ed.jpg)
https://content.homenetiol.com/2001732/2121940/640x480/eca4edf01cce4e98a1c27a791cffeadf.jpg)
https://content.homenetiol.com/2001732/2121940/640x480/bb4d7df003414b06b3f7732c6bc93b52.jpg)
Inventory ID: 7727427
VIN: KL4AMESL4RB160312
New/Used: New
Year: 2024
Make: Buick
Model: Encore GX
Trim: Sport Touring
Transmission: Automatic
Odometer: 3
Color: Cinnabar Metallic
Price: $0 (price not specified)
Features:
Adaptive Cruise Control
Wireless Charging
Audio System - 11" Diagonal HD Color Touchscreen
Advanced Technology Package
Engine - ECOTEC 1.3L Turbo
Rear Park Assist
Cinnabar Metallic
Transmission - 9-Speed Automatic
Axle - 3.17 Final Drive Ratio
HD Surround Vision
Seats - Front Bucket
Ebony Seats with Ebony Interior Accents
Tires - 225/55R18 All-Season, Blackwall
Sport Touring Preferred Equipment Group
Wheels - 18" Gloss Black Aluminum
Lane Departure Warning
Lane Keeping Assist
Front Collision Mitigation
Front Collision Warning
Automatic Highbeams
Turbocharged
Immobilizer
Keyless Start
All Wheel Drive
Power Steering
ABS
4-Wheel Disc Brakes
Aluminum Wheels
Tires - Front Performance
Tires - Rear Performance
Temporary Spare Tire
Automatic Headlights
Heated Mirrors
Power Mirrors
Privacy Glass
Intermittent Wipers
AM/FM Stereo
Bluetooth Connection
Smart Device Integration
Satellite Radio
WiFi Hotspot
Bucket Seats
Premium Synthetic Seats
Pass-Through Rear Seat
Rear Bench Seat
Floor Mats
Adjustable Steering Wheel
Cruise Control
Steering Wheel Audio Controls
Power Windows
Power Door Locks
Keyless Entry
Security System
MP3 Player
Auxiliary Audio Input
A/C
Rear A/C
Rear Defrost
Driver Vanity Mirror
Passenger Vanity Mirror
Driver Illuminated Vanity Mirror
Passenger Illuminated Visor Mirror
Cargo Shade
Traction Control
Stability Control
Daytime Running Lights
Driver Air Bag
Passenger Air Bag
Front Side Air Bag
Rear Side Air Bag
Front Head Air Bag
Rear Head Air Bag
Knee Air Bag
Passenger Air Bag Sensor
Telematics
Back-Up Camera
Blind Spot Monitor
Cross-Traffic Alert
Child Safety Locks
Driver Restriction Features
Tire Pressure Monitor
Photos:
https://content.homenetiol.com/2001732/2121940/640x480/1821cbc4749f4c498f534d508ea6148a.jpg)
https://content.homenetiol.com/2001732/2121940/640x480/f168d7d38e954625828caa495115a9a5.jpg)
https://content.homenetiol.com/2001732/2121940/640x480/6d03ce005d744980af29487c501a5430.jpg)
Inventory ID: 7974856
VIN: KL4AMDSL1RB176123
New/Used: New
Year: 2024
Make: Buick
Model: Encore GX
Trim: Sport Touring
Transmission: Variable
Odometer: 5
Color: Cinnabar Metallic
Price: $32,646
Features:
Adaptive Cruise Control
Wheels - 19" Gloss Black Aluminum
Wireless Charging
Experience Buick Package
Heated Driver and Front Passenger Seats
Remote Vehicle Starter System
Comfort Package
Armrest - Rear Center
Audio System - 11" Diagonal HD Color Touchscreen
Advanced Technology Package
Seat Adjuster - 2-Way Power Driver Lumbar Control
Driver 8-Way Power Seat Adjuster
Interior Protection Package
Transmission - Continuously Variable
Cargo Liner
Engine - ECOTEC 1.3L Turbo
Rear Park Assist
Bose Premium 7-Speaker System
Cinnabar Metallic
All-Weather Floor Liners
Front Passenger Flat-Folding Seatback
Moonroof - Panoramic Tilt-Sliding
Sport Pedal Kit
HD Surround Vision
Seats - Front Bucket
Ebony Seats with Ebony Interior Accents
Tires - 245/45R19 All-Season, Blackwall
Sport Touring Preferred Equipment Group
Steering Wheel - Heated
Lane Departure Warning
Lane Keeping Assist
Front Collision Mitigation
Front Collision Warning
Automatic Highbeams
Turbocharged
Immobilizer
Keyless Start
Front Wheel Drive
Power Steering
ABS
4-Wheel Disc Brakes
Aluminum Wheels
Tires - Front Performance
Tires - Rear Performance
Temporary Spare Tire
Automatic Headlights
Heated Mirrors
Power Mirrors
Privacy Glass
Intermittent Wipers
AM/FM Stereo
Bluetooth Connection
Smart Device Integration
Satellite Radio
WiFi Hotspot
Bucket Seats
Premium Synthetic Seats
Pass-Through Rear Seat
Rear Bench Seat
Floor Mats
Adjustable Steering Wheel
Cruise Control
Steering Wheel Audio Controls
Power Windows
Power Door Locks
Keyless Entry
Security System
MP3 Player
Auxiliary Audio Input
A/C
Rear A/C
Rear Defrost
Driver Vanity Mirror
Passenger Vanity Mirror
Driver Illuminated Vanity Mirror
Passenger Illuminated Visor Mirror
Cargo Shade
Traction Control
Stability Control
Daytime Running Lights
Driver Air Bag
Passenger Air Bag
Front Side Air Bag
Rear Side Air Bag
Front Head Air Bag
Rear Head Air Bag
Knee Air Bag
Passenger Air Bag Sensor
Telematics
Back-Up Camera
Blind Spot Monitor
Cross-Traffic Alert
Child Safety Locks
Driver Restriction Features
Tire Pressure Monitor
Photos:
https://content.homenetiol.com/2001732/212194
Here are the remaining details for the provided inventory:
Inventory ID: 8007405
VIN: KL4AMDSL1RB181564
New/Used: New
Year: 2024
Make: Buick
Model: Encore GX
Trim: Sport Touring
Transmission: Variable
Odometer: 3
Color: Ebony Twilight Metallic
Price: $30,451
Features:
Adaptive Cruise Control
Wireless Charging
Heated Driver and Front Passenger Seats
Remote Vehicle Starter System
Comfort Package
Armrest - Rear Center
Audio System - 11" Diagonal HD Color Touchscreen
Advanced Technology Package
Seat Adjuster - 2-Way Power Driver Lumbar Control
Driver 8-Way Power Seat Adjuster
Interior Protection Package
Cargo Liner
Transmission - Continuously Variable
Engine - ECOTEC 1.3L Turbo
Rear Park Assist
All-Weather Floor Liners
Front Passenger Flat-Folding Seatback
Sport Pedal Kit
HD Surround Vision
Seats - Front Bucket
Ebony Seats with Ebony Interior Accents
Tires - 225/55R18 All-Season, Blackwall
Sport Touring Preferred Equipment Group
Steering Wheel - Heated
Lane Departure Warning
Lane Keeping Assist
Front Collision Mitigation
Front Collision Warning
Automatic Highbeams
Turbocharged
Immobilizer
Keyless Start
Front Wheel Drive
Power Steering
ABS
4-Wheel Disc Brakes
Aluminum Wheels
Tires - Front Performance
Tires - Rear Performance
Temporary Spare Tire
Automatic Headlights
Heated Mirrors
Power Mirrors
Privacy Glass
Intermittent Wipers
AM/FM Stereo
Bluetooth Connection
Smart Device Integration
Satellite Radio
WiFi Hotspot
Bucket Seats
Premium Synthetic Seats
Pass-Through Rear Seat
Rear Bench Seat
Floor Mats
Adjustable Steering Wheel
Cruise Control
Steering Wheel Audio Controls
Power Windows
Power Door Locks
Keyless Entry
Security System
MP3 Player
Auxiliary Audio Input
A/C
Rear A/C
Rear Defrost
Driver Vanity Mirror
Passenger Vanity Mirror
Driver Illuminated Vanity Mirror
Passenger Illuminated Visor Mirror
Cargo Shade
Traction Control
Stability Control
Daytime Running Lights
Driver Air Bag
Passenger Air Bag
Front Side Air Bag
Rear Side Air Bag
Front Head Air Bag
Rear Head Air Bag
Knee Air Bag
Passenger Air Bag Sensor
Telematics
Back-Up Camera
Blind Spot Monitor
Cross-Traffic Alert
Child Safety Locks
Driver Restriction Features
Tire Pressure Monitor
Photos:
https://content.homenetiol.com/2001732/2121940/640x480/97fbd904ae0b4a91998ff5ff21c4efc3.jpg)
https://content.homenetiol.com/2001732/2121940/640x480/3d60b87948e0406cab832978ccb68d9b.jpg)
https://content.homenetiol.com/2001732/2121940/640x480/2bc61680531e40ec8a53dfa2855d86fe.jpg)
Inventory ID: 7897638
VIN: KL4AMESL3RB162553
New/Used: New
Year: 2024
Make: Buick
Model: Encore GX
Trim: Sport Touring
Transmission: Automatic
Odometer: 3
Color: Moonstone Gray Metallic
Price: $0 (price not specified)
Features:
Adaptive Cruise Control
Wireless Charging
Audio System - 11" Diagonal HD Color Touchscreen
Advanced Technology Package
Engine - ECOTEC 1.3L Turbo
Rear Park Assist
Moonstone Gray Metallic
Transmission - 9-Speed Automatic
Axle - 3.17 Final Drive Ratio
HD Surround Vision
Seats - Front Bucket
Ebony Seats with Ebony Interior Accents
Tires - 225/55R18 All-Season, Blackwall
Sport Touring Preferred Equipment Group
Wheels - 18" Gloss Black Aluminum
Lane Departure Warning
Lane Keeping Assist
Front Collision Mitigation
Front Collision Warning
Automatic Highbeams
Turbocharged
Immobilizer
Keyless Start
All Wheel Drive
Power Steering
ABS
4-Wheel Disc Brakes
Aluminum Wheels
Tires - Front Performance
Tires - Rear Performance
Temporary Spare Tire
Automatic Headlights
Heated Mirrors
Power Mirrors
Privacy Glass
Intermittent Wipers
AM/FM Stereo
Bluetooth Connection
Smart Device Integration
Satellite Radio
WiFi Hotspot
Bucket Seats
Premium Synthetic Seats
Pass-Through Rear Seat
Rear Bench Seat
Floor Mats
Adjustable Steering Wheel
Cruise Control
Steering Wheel Audio Controls
Power Windows
Power Door Locks
Keyless Entry
Security System
MP3 Player
Auxiliary Audio Input
A/C
Rear A/C
Rear Defrost
Driver Vanity Mirror
Passenger Vanity Mirror
Driver Illuminated Vanity Mirror
Passenger Illuminated Visor Mirror
Cargo Shade
Traction Control
Stability Control
Daytime Running Lights
Driver Air Bag
Passenger Air Bag
Front Side Air Bag
Rear Side Air Bag
Front Head Air Bag
Rear Head Air Bag
Knee Air Bag
Passenger Air Bag Sensor
Telematics
Back-Up Camera
Blind Spot Monitor
Cross-Traffic Alert
Child Safety Locks
Driver Restriction Features
Tire Pressure Monitor
Photos:
https://content.homenetiol.com/2001732/2121940/640x480/896e33d334384f03b387dde342ef0eba.jpg)
https://content.homenetiol.com/2001732/2121940/640x480/6fea58f15f9543539226869195b98d6b.jpg)
https://content.homenetiol.com/2001732/2121940/640x480/44986fcf656e41e1b312e843050e3ac4.jpg)
Inventory ID: 8007403
VIN: KL4AMDSL5RB182801
New/Used: New
Year: 2024
Make: Buick
Model: Encore GX
Trim: Sport Touring
Transmission: Variable
Odometer: 4
Color: Summit White
Price: $28,661
Features:
Adaptive Cruise Control
Wireless Charging
Audio System - 11" Diagonal HD Color Touchscreen
Advanced Technology Package
Interior Protection Package
Transmission - Continuously Variable
Cargo Liner
Engine - ECOTEC 1.3L Turbo
Rear Park Assist
All-Weather Floor Liners
Sport Pedal Kit
HD Surround Vision
Seats - Front Bucket
Ebony Seats with Ebony Interior Accents
Tires - 225/55R18 All-Season, Blackwall
Sport Touring Preferred Equipment Group
Steering Wheel - Heated
Lane Departure Warning
Lane Keeping Assist
Front Collision Mitigation
Front Collision Warning
Automatic Highbeams
Turbocharged
Immobilizer
Keyless Start
Front Wheel Drive
Power Steering
ABS
4-Wheel Disc Brakes
Aluminum Wheels
Tires - Front Performance
Tires - Rear Performance
Temporary Spare Tire
Automatic Headlights
Heated Mirrors
Power Mirrors
Privacy Glass
Intermittent Wipers
AM/FM Stereo
Bluetooth Connection
Smart Device Integration
Satellite Radio
WiFi Hotspot
Bucket Seats
Premium Synthetic Seats
Pass-Through Rear Seat
Rear Bench Seat
Floor Mats
Adjustable Steering Wheel
Cruise Control
Steering Wheel Audio Controls
Power Windows
Power Door Locks
Keyless Entry
Security System
MP3 Player
Auxiliary Audio Input
A/C
Rear A/C
Rear Defrost
Driver Vanity Mirror
Passenger Vanity Mirror
Driver Illuminated Vanity Mirror
Passenger Illuminated Visor Mirror
Cargo Shade
Traction Control
Stability Control
Daytime Running Lights
Driver Air Bag
Passenger Air Bag
Front Side Air Bag
Rear Side Air Bag
Front Head Air Bag
Rear Head Air Bag
Knee Air Bag
Passenger Air Bag Sensor
Telematics
Back-Up Camera
Blind Spot Monitor
Cross-Traffic Alert
Child Safety Locks
Driver Restriction Features
Tire Pressure Monitor
Photos:
https://content.homenetiol.com/2001732/2121940/640x480/97fbd904ae0b4a91998ff5ff21c4efc3.jpg)
https://content.homenetiol.com/2001732/2121940/640x480/3d60b87948e0406cab832978ccb68d9b.jpg)
https://content.homenetiol.com/2001732/2121940/640x480/2bc61680531e40ec8a53dfa2855d86fe.jpg)





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
