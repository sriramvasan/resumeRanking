from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from typing import List
import textract
import json
import os 
from dotenv import load_dotenv
from pydantic import BaseModel
from IPython.display import display , Markdown

load_dotenv()

app = FastAPI()
# OpenAI API Key setup
api_key = os.getenv("OPEN_AI_API")
client = OpenAI(api_key = api_key)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)




def extract_text_from_file(file_path: str) -> str:
    """Extracts text from a given file."""
    text = textract.process(file_path)
    return text.decode('utf-8')

class CriteriaExtractionResponse(BaseModel):
    criteria: List[str]

@app.post("/extract-criteria" , response_model=CriteriaExtractionResponse , summary="Extract Job Criteria",
          description="Extracts key ranking criteria from an uploaded job description file (PDF or DOCX)." )
async def extract_criteria(file: UploadFile = File(...)) -> CriteriaExtractionResponse:
    """This endpoint generates the key criteria for a job description. 

    Args:
        file (UploadFile): A PDF file containing the description of a job.

    Returns:
        dict: a list of important criteria for the job
    """
    if not file.filename.endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Save the file temporarily (could use a temp file as well)
    temp_file_path = f"temp/{file.filename}"
    with open(temp_file_path, 'wb+') as f:
        f.write(await file.read())

    # Helps to extract text from the file
    extracted_text = extract_text_from_file(temp_file_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system","content": "You are an experienced Hiring Manager. For the given Job description, look for the necessary criteria that is crucial for the role."},
            {"role": "user", "content": f"Extract key ranking criteria from the following job description: {extracted_text}, do not separate into multiple categories, return the json as a list of criterias"}
        ],
        
        response_format={
    "type": "json_schema",
    "json_schema": {
      "name": "criteria_list",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "criteria": {
            "type": "array",
            "description": "A list of criteria as strings.",
            "items": {
              "type": "string"
            }
          }
        },
        "required": [
          "criteria"
        ],
        "additionalProperties": False
      }
    }
  },
        temperature=0.5,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    
    criteria = response.choices[0].message.content
    
    criteria_dict = json.loads(criteria)
    
    return CriteriaExtractionResponse(**x)
    
 
