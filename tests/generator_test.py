from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

from core.config import settings

client = genai.Client(api_key=settings.GOOGLE_API_KEY)

prompt = "Generate a fantasy character portrait of a warrior with long hair and a beard, wearing armor, in a dramatic style."

response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents=[prompt],
)

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save(
            "/Users/marek.lipan/Desktop/Projects/generated-adventures/core/generated_image.png"
        )
