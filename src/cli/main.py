import typer
import asyncio
from src.ingestion.pdf import convert_pdf_to_markdown, test_crop_pdf_page
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

app = typer.Typer(name="epinoia")

@app.command()
def main(pdf_url: str, top: float = 0, bottom: float = 0):
    """
    Send a message to the AI and get a response.
    """
    asyncio.run(convert_pdf_to_markdown(pdf_url, top_crop_percent=top, bottom_crop_percent=bottom))

@app.command()
def crop_pdf_page(pdf_url: str, top_percent: float, bottom_percent: float):
    """
    Test the crop_pdf_page function.
    """
    print(f"Cropping PDF page {pdf_url} with top crop {top_percent} and bottom crop {bottom_percent}")
    asyncio.run(test_crop_pdf_page(pdf_url, top_percent, bottom_percent))

if __name__ == "__main__":
    app()