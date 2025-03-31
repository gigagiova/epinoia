from .base import LLM
from google import genai
import os
from google.genai.types import File
from typing import List
import asyncio


class GeminiModel(LLM):
    
    @property
    def client(self) -> genai.Client:
        api_key = os.getenv("GEMINI_API_KEY")
        return genai.Client(api_key=api_key)

    async def run(self, prompt: str, *, pdf_bytes: List[bytes] = None) -> str:

        contents = [prompt]

        # Converts each pdf bytes in a format that can be used by the client
        for pdf in pdf_bytes or []:
            pdf_file = genai.types.Part.from_bytes(data=pdf, mime_type='application/pdf')
            contents.append(pdf_file)

        # Tries 3 times to get a response from the model
        for i in range(8):
            
            # Formats the request and sends it to the model
            response = self.client.models.generate_content(model=self.model_name, contents=contents)

            try:
                # Extract just the text content from the response
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.text:
                            return part.text
                        
            except Exception as e:
                print(f"Error extracting text from response: {e}")
            
            # If the response is empty, or there is an error, wait and try again
            await asyncio.sleep(1.8**i)

    def upload_file(self, file_path: str, *, file_name: str) -> File:
        return self.client.files.upload(file_path, config={"display_name": file_name})


gemini_2flash = GeminiModel('gemini-2.0-flash')
