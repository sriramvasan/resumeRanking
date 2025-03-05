from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Body , Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from typing import List
import textract
import json
import os 
from dotenv import load_dotenv
from pydantic import BaseModel , validator
from IPython.display import display , Markdown
import pandas as pd
import io
import glob

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

# Clearing the files stored for processing 
def clear_tempfiles(directory:str)-> None:
  """Deletes all the temporary files saved in the 
  """
  # Checking if the directory exists
  if not os.path.exists(directory):
    print(f"Directory {directory} does not exist.")
    return

  # goes through all the files in the directory
  files = glob.glob(os.path.join(directory, '*'))
  for file in files:
    try:
      os.remove(file)
      print(f"Deleted {file}")
    except Exception as e:
      print(f"Error deleting {file}: {str(e)}")

# Saves the file temporarily to be processed or extracted
async def save_tempfile(file: UploadFile) -> None:
  """File is saved to extract or process it later

  Args:
      file (UploadFile): The file that needs to be saved in a temporary location
  """
  if not os.path.exists("temp/"):
      os.mkdir('temp/')
  temp_file_path = f"temp/{file.filename}"
  with open(temp_file_path, 'wb+') as f:
    f.write(await file.read())

def extract_text_from_file(file_path: str) -> str:
  """Extracts text from a given file.

  Args:
      file_path (str): The file path for the file which needs to be extracted

  Returns:
      str: Returns the extracted texts from the file 
  """
  text = textract.process(file_path)
  return text.decode('utf-8')

class CriteriaExtractionResponse(BaseModel):
    criteria: List[str]

@app.post("/extract-criteria" , response_model=CriteriaExtractionResponse , summary="Extract Job Criteria",
          description="Extracts key ranking criteria from an uploaded job description file (PDF or DOCX)." )
async def extract_criteria(file: UploadFile = File(...)) -> CriteriaExtractionResponse:
    """This endpoint extracts the key criteria for a given job description using openAI's LLM . 

    Args:
        file (UploadFile): A PDF file containing the description of a job.

    Returns:
        dict: a list of important criteria for the job
    """
    if not file.filename.endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Save the file temporarily
    temp_file_path = f"temp/{file.filename}"
    await save_tempfile(file)

    # Helps to extract text from the file
    extracted_text = extract_text_from_file(temp_file_path)

    # openai-response
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system","content": "You are an experienced Hiring Manager. For the given Job description, look for the necessary criteria that is crucial for the role."},
            {"role": "user", "content": f"Extract key ranking criteria such as skills, certifications, experience, and qualifications from the following job description: {extracted_text}, do not separate into multiple categories, return the json as a list of criterias. Make the criterias short, 2-3 words"}
        ],
        
        response_format={
    "type": "json_schema",
    "json_schema": {
      "name": "extract-criteria",
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
        temperature=0,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    
    criteria = response.choices[0].message.content

    # deleting the temporarily saved file
    clear_tempfiles('temp/')
    return CriteriaExtractionResponse.parse_raw(criteria)
    

@app.post("/score-resumes", summary="Score Resumes",
          description="Scores multiple resumes based on provided criteria and returns scores in a CSV format.")
async def score_resumes(criteria: str = Form(), files: List[UploadFile] = File(...)) -> StreamingResponse:
  """This endpoint scores uploaded resumes based on the provided job criteria using OpenAI's LLM.

  Args:
      request (criteria , files): Contains the list of criteria and uploaded resume files.

  Returns:
      StreamingResponse: A CSV file containing the scores for each candidate.
  """
  
  try:
    criteria_json = json.loads(criteria)
    criteria = CriteriaExtractionResponse(**criteria_json)
  except json.JSONDecodeError:
    raise HTTPException(status_code=400, detail="Invalid JSON format")
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))

  results = []

  for file in files:
    temp_file_path = f"temp/{file.filename}"
    await save_tempfile(file)
    
    extracted_text = extract_text_from_file(temp_file_path)
    
    individual_scores = []
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an experienced Hiring Manager.Match the text against the provided criteria and score based on the presence and relevance of each of the criteria. The value for each of the criteria should be between 0-5. Keep Track of the name for each of the candidates, don't add any explanation, just the value."},
            {"role": "user", "content": str(criteria)},
            {"role": "user", "content": extracted_text}
        ],
        response_format={
          "type": "json_schema",
          "json_schema": {
            "name": "scores_list",
            "strict": True,
            "schema": {
              "type": "object",
              "properties": {
                "name": {
                  "type": "string",
                  "description": "The name associated with the scores."
                },
                "scores": {
                  "type": "array",
                  "description": "A list of numerical scores.",
                  "items": {
                    "type": "number",
                    "description": "A single score in the list."
                  }
                }
              },
              "required": [
                "name",
                "scores"
              ],
              "additionalProperties": False
            }
          }
        },
        temperature=0,
        max_completion_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    res = response.choices[0].message.content
    res = json.loads(res)

    if len(criteria.criteria) != len(res['scores']):
      raise ValueError("The number of criteria must match the number of scores")

    criteria_scores = dict(zip(criteria.criteria,res['scores']))

    total_scores = sum(res['scores'])

    individual_scores = {
      "name" : res["name"],
      **criteria_scores,
      "total_scores": total_scores
    }

    results.append(individual_scores)

  # Convert results to DataFrame
  df = pd.DataFrame(results)
  df.name = df.name.apply(lambda x: x.title())
  df = df.sort_values(by = ["total_scores"], ascending = False)
  
  # Create a CSV from the DataFrame
  stream = io.StringIO()
  df.to_csv(stream, index=False)
  stream.seek(0)  

  clear_tempfiles('temp/')
  return StreamingResponse(stream, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=scores.csv"})

            