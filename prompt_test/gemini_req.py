import requests
import json
import os
from dotenv import load_dotenv
from answer import transcribe_audio, record_audio

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
API_KEY = os.getenv("GEMINI_API_KEY")

# File where JSON data will be persisted (in current directory)
JSON_FILE = "medical_form.json"

# Load the JSON from file if it exists
def load_json():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as file:
            return json.load(file)
    else:
        return {
            "patient_id": "",
            "name": {
                "first_name": "",
                "last_name": ""
            },
            "date_of_birth": "",
            "gender": "",
            "contact_information": {
                "email": "",
                "phone_number": ""
            },
            "address": {
                "street": "",
                "city": "",
                "state": "",
                "zip_code": ""
            },
            "medical_history": {
                "allergies": [],
                "current_medications": [],
                "past_surgeries": [],
                "chronic_conditions": []
            },
            "primary_physician": {
                "name": "",
                "contact_number": ""
            },
            "emergency_contact": {
                "name": "",
                "relationship": "",
                "contact_number": ""
            },
            "insurance_information": {
                "provider_name": "",
                "policy_number": "",
                "group_number": ""
            },
            "date_of_last_visit": "",
            "reason_for_visit": "",
            "symptoms": [],
            "diagnosis": "",
            "treatment_plan": {
                "medications_prescribed": [],
                "procedures_recommended": []
            }
        }

# Save the JSON to file
def save_json(data):
    try:
        with open(JSON_FILE, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"JSON saved successfully in the current directory as {JSON_FILE}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

# Function to interact with Google Gemini
def run_gemini(prompt):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(f"{url}?key={API_KEY}", headers=headers, json=data)

    if response.status_code == 200:
        gemini_response = response.json()
        print("Response JSON:", gemini_response)
        
        # Return the response for further processing
        return gemini_response
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")
        return None

# Function to clean and extract the JSON string from Gemini output
def extract_json(gemini_response):
    # Check if there are candidates in the response
    if 'candidates' in gemini_response and len(gemini_response['candidates']) > 0:
        # Extract the first candidate's content
        content = gemini_response['candidates'][0]['content']
        if 'parts' in content and len(content['parts']) > 0:
            # Get the text from the first part
            response_text = content['parts'][0]['text']
            # Remove the code formatting and parse the JSON
            json_text = response_text.strip('```json\n').strip('```').strip()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"Failed to decode extracted JSON: {e}")
                return None
    else:
        print("No valid 'candidates' found in response.")
        return None

# Example usage
def main():
    print(f"Current working directory: {os.getcwd()}")
    medical_form = load_json()

    # Example usage of record_audio and transcribe_audio functions
    wav_path = record_audio()  # Assume this function exists
    transcribed_text = transcribe_audio(wav_path)  # Assume this function exists
    print(transcribed_text)

    prompt = f"""Based on the following transcribed text from a doctor's notes, update only the relevant fields of the JSON object below with the information provided in the text. Do not autofill or guess any information:

    Transcribed text: 
    \"\"\"{transcribed_text}\"\"\"

    Existing JSON object to update:

    {json.dumps(medical_form, indent=4)}
    """

    # Get the output from Google Gemini
    gemini_response = run_gemini(prompt)

    if gemini_response:
        # Extract the JSON from the Gemini response
        extracted_json = extract_json(gemini_response)

        if extracted_json:
            # Merge the extracted fields with the existing JSON
            for key, value in extracted_json.items():
                if isinstance(value, dict):
                    # Update nested dictionaries
                    medical_form[key].update(value)
                else:
                    medical_form[key] = value

            # Save the updated JSON to the file
            save_json(medical_form)
            print("Updated medical form saved successfully.")
        else:
            print("No valid JSON extracted from Gemini output.")
    else:
        print("No output from Gemini.")

main()


