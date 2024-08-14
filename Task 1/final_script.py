import os
import pytesseract
from PIL import Image
import numpy as np
import docx
import fitz
import logging
import cv2
import io

logging.basicConfig(filename="text_extraction.log", level=logging.INFO, format='%(asctime)s %(message)s')

# Path to Tesseract library 
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\tr4\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

class TextExtractionTool:

    def __init__(self, input_folder):
        self.input_folder = input_folder

    def file_type(self, file_path):
        # Function to extract the file type
        _, file_extension = os.path.splitext(file_path)
        return file_extension.lower()

    def extract_from_image(self, file_path):
        # Function to extract the texts from Images 
        image = cv2.imread(file_path)
        # Preprocessing steps for the Image 
        norm_img = np.zeros((image.shape[0], image.shape[1]))
        image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2,2),np.uint8)
        image = cv2.erode(image, kernel, iterations=1)

        logging.info(f"Extracting text from image: {file_path}")
        text = pytesseract.image_to_string(image)
        return text

    def extract_from_pdf(self, file_path):
        # Function to extract the texts from the PDF
        logging.info(f"Extracting text from PDF: {file_path}")
        doc = fitz.open(file_path)
        text = ""
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
            # If Images present in the PDf, load and extract the Images 
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)
        for img in images:
            logging.info(f"Total number of images found: {len(images)}, - File Name-{file_path}")
            logging.info(f"Extracting text from image in PDF: {file_path}")
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text:
                text += "\n" + ocr_text
            else:
                text += ""
                logging.error(f"Unable to Extract image from {file_path}")
        return text

    def extract_from_docx(self, file_path):
        # Function to extract the texts from the Document 
        logging.info(f"Extracting text from document: {file_path}")
        doc = docx.Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs])
        images = []
        for rel in doc.part.rels:
            if "image" in doc.part.rels[rel].target_ref:
                image = doc.part.rels[rel].target_part.blob
                img = Image.open(io.BytesIO(image))
                images.append(img)
        for img in images:
            logging.info(f"Total number of images found:  {len(images)}, - File Name-{file_path}")
            logging.info(f"Extracting text from image in document: {file_path}")
            ocr_text = pytesseract.image_to_string(img)
            # ocr_text = self.extract_from_image(self, file_path)
            if ocr_text:
                text += "\n" + ocr_text
            else:
                text += ""
                logging.error(f"Unable to Extract image from {file_path}")
        return text

    def extract_from_txt(self, file_path):
        # Function to extract the texts from the Text File
        logging.info(f"Extracting text from text file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def valid_characters(self, text):
        # Function to Validate Characters 
        try:
            text.encode('utf-8')
            return True
        except UnicodeEncodeError:
            return False

    def process_file(self, file_path):
        # Base function linker 
        file_name = os.path.basename(file_path)
        file_extension = self.file_type(file_path)
        if file_extension in [".jpg, .pdf., .png"]:
            print('''So basically what should I be doing, If I go tomorrow to Gonikoppal and get the sign done
                  then I shall have the amount ready, I should also be checking on the documents required to send and also I should get it notarised 
                  I should also check if I need any additional documents, since tmr is the last date monday or so I can make the transaction, and send the documents visa post
                  ''')

        if file_extension in ['.jpg', '.jpeg', '.png']:
            extracted_text = self.extract_from_image(file_path)
        elif file_extension == '.pdf':
            extracted_text = self.extract_from_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            extracted_text = self.extract_from_docx(file_path)
        elif file_extension == '.txt':
            extracted_text = self.extract_from_txt(file_path)
        else:
            logging.error(f"Unsupported file type: {file_extension} - {file_name}")
            return file_name, None

        if extracted_text and not self.valid_characters(extracted_text):
            logging.warning(f"Extracted text contains invalid characters: {file_name}")

        return file_name, extracted_text

    def process_folder(self):
        # Function to create an Output folder and update it
        output_folder = "Output"
        os.makedirs(output_folder, exist_ok=True)
        logging.info("Output folder created")
        files = os.listdir(self.input_folder)
        for file_name in files:
            file_path = os.path.join(self.input_folder, file_name)
            if os.path.isfile(file_path):
                original_name, extracted_text = self.process_file(file_path)
                if extracted_text:
                    output_file_name = f"{os.path.splitext(original_name)[0]}_extracted.txt"
                    output_file_path = os.path.join(output_folder, output_file_name)
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)
                    print(f"Extracted text saved to: {output_file_path}")
                else:
                    print(f"Extraction failed for: {file_path}")


if __name__ == "__main__":
    input_folder = "Input"
    tool = TextExtractionTool(input_folder)
    tool.process_folder()
