import requests
import json
import re
import os
from answer import transcribe_audio, record_audio

# File where JSON data will be persisted (in current directory)
JSON_FILE = "medical_form.json"

# Load the JSON from file if it exists
def load_json():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as file:
            return json.load(file)
    else:
        # Return an empty JSON structure if file doesn't exist
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

# Function to interact with LLaMA
def run_llama(prompt):
    url = "http://localhost:11434/v1/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3.1",
        "prompt": prompt
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        llama_response = response.json()
        print("Response JSON:", llama_response)
        
        # Accessing the text under the 'choices' key
        if 'choices' in llama_response and len(llama_response['choices']) > 0:
            return llama_response['choices'][0]['text']
        else:
            return f"No valid 'choices' found in response."
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")
        return None

# Function to clean the JSON string by removing comments and fixing formatting issues
def clean_json(llama_output):
    # Remove comments (e.g., // comment) from the JSON string
    cleaned_output = re.sub(r'//.*', '', llama_output)
    
    # Optionally, remove line breaks within the JSON if they cause issues
    cleaned_output = re.sub(r'\n', '', cleaned_output)
    
    return cleaned_output

# Updated function to extract JSON from LLaMA output
def extract_json(llama_output):
    # Clean the output first
    cleaned_output = clean_json(llama_output)
    
    # Use regex to search for the JSON-like object
    json_match = re.search(r'{.*}', cleaned_output, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to decode extracted JSON: {e}")
            return None
    else:
        print("No JSON found in LLaMA output.")
        return None

# Example usage
def main():
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Load existing JSON data
    medical_form = load_json()

    wav_path = record_audio()
    transcribed_text = transcribe_audio(wav_path)
    print(transcribed_text)

    # Insert the transcribed text and existing JSON into the prompt
    prompt = f"""Based on the following transcribed text from a doctor's notes, update only the relevant fields of the JSON object below with the information provided in the text. Do not autofill or guess any information:

    Transcribed text: 
    \"\"\"{transcribed_text}\"\"\"

    Existing JSON object to update:

    {json.dumps(medical_form, indent=4)}
    """

    # Get the output from LLaMA
    llama_output = run_llama(prompt)

    if llama_output:
        # Extract the JSON from the LLaMA output
        extracted_json = extract_json(llama_output)

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
            print("No valid JSON extracted from LLaMA output.")
    else:
        print("No output from LLaMA.")

main()



