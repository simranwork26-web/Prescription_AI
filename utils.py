import base64
import requests
import os
from dotenv import load_dotenv

# Load env file
load_dotenv()

# FIRST define key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# THEN print
print("API KEY:", OPENAI_API_KEY)


def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

def extract_medicines(base64_image):
    import requests

    url = "https://api.openai.com/v1/responses"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4.1-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract medicines from this prescription and return JSON."
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)

    result = response.json()

    try:
        return result["output"][0]["content"][0]["text"]
    except:
        return None