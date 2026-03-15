import pytesseract
from PIL import Image

def extract_text_from_image(image_file):
    """
    Extracts text from an uploaded image file using Tesseract.
    """
    try:
        # Load the image using Pillow
        img = Image.open(image_file)
        
        # Windows: You may need to specify the Tesseract EXE path:
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Use pytesseract to convert image to string
        text = pytesseract.image_to_string(img)
        return text.strip() if text.strip() else "Could not read any text."
    except Exception as e:
        return f"Error processing image: {str(e)}"