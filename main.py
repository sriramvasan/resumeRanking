from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
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
            {"role": "user", "content": f"Extract key ranking criteria from the following job description: {extracted_text}, do not separate into multiple categories, return the json as a list of criterias. Make the criterias short, 2-3 words"}
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
    criteria_dict = CriteriaExtractionResponse.parse_raw(criteria)


    criteria_dict = json.loads(criteria)
    
    # session_id = uuid.uuid4().hex  # generate a unique session ID
    # criteria_storage[session_id] = criteria_dict
    # return {"session_id": session_id, "criteria": criteria_dict}

    # return CriteriaExtractionResponse(**criteria_dict)

    return criteria_dict
    

class ResumeScoringRequest(BaseModel):
    criteria: List[str]
    files: List[UploadFile]


@app.post("/score-resumes", summary="Score Resumes",
          description="Scores multiple resumes based on provided criteria and returns scores in a CSV format.")
async def score_resumes(criteria: str, files: List[UploadFile] = File(...)) -> StreamingResponse:
# async def score_resumes(session_id: str, files: List[UploadFile] = File(...)) -> StreamingResponse:
# async def score_resumes(ResumeScoringRequest) -> StreamingResponse:
  """This endpoint scores uploaded resumes based on the provided job criteria using OpenAI's LLM.

  Args:
      request (ResumeScoringRequest): Contains the list of criteria and uploaded resume files.

  Returns:
      StreamingResponse: A CSV file containing the scores for each candidate.
  """
  # if session_id not in criteria_storage:
  #     raise HTTPException(status_code=404, detail="Session ID not found")
  # criteria = criteria_storage[session_id]
  
  results = []

  criteria = CriteriaExtractionResponse.parse_raw(criteria)

  for file in files:
    temp_file_path = f"temp/{file.filename}"
    with open(temp_file_path, 'wb+') as f:
        f.write(await file.read())
    
    extracted_text = extract_text_from_file(temp_file_path)
    
    individual_scores = []
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an experienced Hiring Manager. Score the relevance of the resume text to the following set of criterias. The value for each of the criteria should be between 0-5.Keep Track the name for each of the candidates, don't add any explanation, just the value."},
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
    # res = dict(res)
    # print(res, type(res), res.keys(), res.values(), res['scores'])

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
  
  # Create a CSV from the DataFrame
  stream = io.StringIO()
  df.to_csv(stream, index=False)
  stream.seek(0)  # Go back to the start of the StringIO object

  # Return StreamingResponse
  return StreamingResponse(stream, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=scores.csv"})
  # return results
            