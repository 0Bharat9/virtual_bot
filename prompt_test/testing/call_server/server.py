from flask import Flask, request, session
from flask_socketio import SocketIO, disconnect
from flask_cors import CORS
import random
import base64
import os
import json
from datetime import datetime, timedelta
import bcrypt
import ffmpeg
import whisper
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from helperFunctions import (
    system_prompt,
    generate_random_string,
    send_sms,
    run_gemini,
)
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.prompts import (
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

load_dotenv()

uri = "mongodb+srv://kdivu03:ABCD39@cluster0.ekbsw.mongodb.net"
client = MongoClient(uri)
collection = client["nimhans"]

user = collection["users"]
doctor = collection["doctors"]
appointment = collection["appointments"]


# Initialize Flask app, CORS, and SocketIO
app = Flask(_name_)
app.config["SECRET_KEY"] = "secret!"
# CORS(app, origins="http://localhost:5173")
# socketio = SocketIO(app, cors_allowed_origins="http://localhost:5173")
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*")


# Define directory for saving audio files
AUDIO_SAVE_PATH = "saved_audio"
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

# Load Whisper model
model = whisper.load_model("base")


@app.route("/", methods=["GET"])
def hello_world():
    return "Hello, World!"


@socketio.on("connect")
def handle_connect():
    session["memory"] = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    session["responses"] = []
    session["llm"] = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    session["prompt"] = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )
    session["conversation"] = LLMChain(
        llm=session["llm"], prompt=session["prompt"], memory=session["memory"]
    )
    response = {
        "message": "Welcome to the National Institute of Mental Health and Neurosciences Call appointment Service. You can now book your appointment while talking with our AI bot Alex. ",
        "continue_conversation": True,
    }
    socketio.emit("speak", response, to=request.sid)


@socketio.on("phoneNumber")
def handle_register(data):
    session["phoneNumber"] = "+91" + data["phoneNumber"]

    found_user = user.find_one({"phone": session["phoneNumber"]})
    print(found_user)

    if found_user:
        session["user"] = found_user

    else:

        password = generate_random_string(8)
        hashed_password = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")
        user.insert_one(
            {
                "name": "Anonymous",
                "email": "",
                "address": {
                    "line1": "",
                    "line2": "",
                },
                "gender": "Not Selected",
                "password": hashed_password,
                "dob": "Not Selected",
                "phone": session["phoneNumber"],
                "image": "https://cdn.chandanbir.me/default.jpeg",
            }
        )
        # set the pasword to the phone number

        session["user"] = user.find_one({"phone": session["phoneNumber"]})
        res = send_sms(
            "your pass for the web is " + password,
            session["phoneNumber"],
        )
        print(res)


@socketio.on("call")
def handle_call_connect(data):
    continue_conversation = True
    client_sid = request.sid

    audio_base64 = data.get("audio")
    if not audio_base64:
        print("No audio data received.")
        return

    # Decode the base64 audio data
    audio_data = base64.b64decode(audio_base64)

    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    file_path = os.path.join(AUDIO_SAVE_PATH, f"audio_{timestamp}_{client_sid}.webm")

    # Save the decoded audio data to a file
    try:
        with open(file_path, "wb") as audio_file:
            audio_file.write(audio_data)
        # print(f"Audio data saved to {file_path}")
    except Exception as e:
        print(f"Failed to save audio data: {e}")

    # Convert the audio to WAV format for transcription
    output_file = file_path.replace(".webm", ".wav")
    try:
        ffmpeg.input(file_path).output(output_file).run()
    except Exception as e:
        print(f"Failed to convert audio: {e}")
        return

    # Transcribe audio using Whisper
    result = model.transcribe(output_file, language="en")
    transcription_text = result.get("text", "Transcription failed.")
    if not transcription_text:
        transcription_text = "i didn't get that"
    # print(f"Transcription: {transcription_text}")

    session["responses"].append(transcription_text)

    # Add transcription to chat history and get response
    session["memory"].chat_memory.add_user_message(transcription_text)
    response = session["conversation"].invoke({"text": transcription_text})
    response_text = response.get("text", "No response generated.")
    session["memory"].chat_memory.add_ai_message(response_text)

    if (
        "goodbye" or "good bye" or "goodbye." or "good bye." or "bye bye" or "bye bye."
    ) in transcription_text.lower():
        print("good bye found")
        continue_conversation = False

    response = {
        "message": response_text,
        "continue_conversation": continue_conversation,
    }
    socketio.emit("speak", response, to=request.sid)

    if not continue_conversation:
        print("Server disconnected the client")
        disconnect(sid=request.sid)

    # delete the audio file
    os.remove(file_path)
    os.remove(output_file)


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected", request.sid)

    prompt = (
        "Based on the following user responses from a doctor's notes, update the relevant fields of the JSON object. Do not autofill or guess any information:\n\n"
        "user responses:\n"
        + str(session["responses"])
        + "\n\n"
        + "aslo based on the user responses add a a severity (High Medium Low) of illness  field to the json object"
    )

    gemini_response = run_gemini(prompt)
    # parse this to json
    gemini_response = json.loads(gemini_response)

    gemini_response = gemini_response["patient_info"]

    print(gemini_response)

    if "user" in session and session["user"]["name"] == "Anonymous":
        session["user"] = user.update_one(
            {"_id": session["user"]["_id"]},
            {
                "$set": {
                    "name": gemini_response["name"],
                    "age": gemini_response["age"],
                }
            },
        )
        session["user"] = user.find_one({"phone": session["phoneNumber"]})

    # assign a rondom doc to the user appointment

    doctor_emails = [
        "ravindranath@gmail.com",
        "nalini@gmail.com",
        "ravi@gmail.com",
        "pritam@gmail.com",
        "seena@gmail.com",
    ]

    random_index = random.randint(0, 4)
    doc = doctor.find_one({"email": doctor_emails[random_index]})
    print(session["phoneNumber"])

    userObj = user.find_one({"phone": session["phoneNumber"]})

    next_date = (datetime.now() + timedelta(days=1)).strftime("%d_%m_%y")

    random_time = str(random.randint(1, 11)) + ":" + "30"

    app = appointment.insert_one(
        {
            "userId": userObj["_id"],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "userData": session["user"],
            "docId": doc["_id"],
            "docData": doc,
            "amount": doc["fees"],
            "slotDate": next_date,
            "slotTime": random_time,
            "cancelled": False,
            "sverityOfSymptom": gemini_response["sverityOfSymptom"],
            "payment": False,
            "isCompleted": False,
            "symptom": gemini_response["symptom"],
            "durationOfSymptom": gemini_response["durationOfSymptom"],
            "healthInsurance": "no",
        }
    )
    send_sms(
        "your appointment is booked with id " + str(app.inserted_id),
        session["phoneNumber"],
    )

    # send_sms(
    #     "Your request for the appointment recived with id "
    #     + str(app["_id"])
    #     + ". We will get back to you soon",
    #     session["phoneNumber"],
    # )


if _name_ == "_main_":
    socketio.run(app, port=3000, debug=True)
