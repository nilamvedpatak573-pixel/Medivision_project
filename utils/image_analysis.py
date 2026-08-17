import os
import io
import fitz
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# List of models to try if one of them fails 
MODEL_NAMES = [
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro"
]
model = None
for model_name in MODEL_NAMES:
    try:
        model = genai.GenerativeModel(model_name)
        print(f"Using model: {model_name}")
        break
    except Exception:
        continue

if model is None:
    raise Exception("No compatible Gemini model found.")    


# create the function
def analyze_image(uploaded_file):

    # if user uploads an image
    if uploaded_file.type.startswith("image"):

        image = Image.open(uploaded_file)

        try:
            response = model.generate_content(
                [
                    """
                    You are a medical assistant.

                    Analyze this medical report or X-ray briefly.

                    Provide:
                    1. Key findings
                    2. Important abnormalities
                    3. Simple explanation
                    4. General recommendation to consult a doctor

                    Keep the response short.
                    Do not provide a final diagnosis.
                    """,
                    image
                ],
                generation_config={
                    "max_output_tokens": 300,
                    "temperature": 0.3
                }
            )

            return response.text

        except Exception as e:
            return (
                "AI image analysis service is temporarily unavailable.\n\n"
                f"Techniccal Details:\n{e}"
            )
            


    # if user uploads PDF
    elif uploaded_file.type == "application/pdf":

        try:
            pdf = fitz.open(
                stream=uploaded_file.read(),
                filetype="pdf"
            )

            text = ""

            for page in pdf:
                text += page.get_text()

            # Limit text size
            text = text[:2500]

            return text

        except Exception as e:
            return f"Error: {str(e)}"