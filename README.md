# ResumeRanking

This project provides two RESTful API endpoints using FastAPI that automate the process of ranking resumes based on job descriptions. The first endpoint extracts key ranking criteria from job descriptions, while the second scores resumes against those extracted criteria.


## Features
**Extract Criteria:** Extracts key ranking criteria from a job description file.

**Score Resumes:** Scores resumes based on the extracted criteria and returns a CSV file with the scores.


## Technologies
* Python 3.8+
* FastAPI
* OpenAI API
* textract for text extraction


## Setup Instructions
###  Prerequisites
* Python 3.8 or higher
* pip


### Installation

1. Clone the repository:

```https://github.com/sriramvasan/resumeRanking.git
cd resume-ranking-api
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Set up the environment variables:


* Create a .env file in the project root directory.
* Add `OPEN_AI_API=your_openai_api_key_here` to the file.

### Running the Application
To run the application, use the following command:

```
uvicorn main:app --reload
```

This will start the FastAPI application with live reloading enabled.

## API Documentation
After running the application, you can access the Swagger UI documentation at `http://127.0.0.1:8000/docs`. This UI allows you to:

* View detailed documentation of each endpoint.
* Try out the API directly from your browser.

### Endpoints
1. POST /extract-criteria

* Description: Extracts key ranking criteria from a provided job description file.
* Input: Multipart form-data with the key file containing the job description file.

2. POST /score-resumes

* Description: Scores resumes based on provided criteria and returns scores in a CSV format.
* Input: Multipart form-data with the key criteria containing a JSON string of criteria and files containing multiple resume files.

## Usage

Here are some examples of how to use the API with curl:

### Extract Criteria

```curl -X 'POST' \
  'http://127.0.0.1:8000/extract-criteria' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path_to_job_description.pdf'
```

### Score Resumes

```curl -X 'POST' \
  'http://127.0.0.1:8000/score-resumes' \
  -H 'accept: application/octet-stream' \
  -H 'Content-Type: multipart/form-data' \
  -F 'criteria={"criteria":["5+ years experience in Python","Must have certification XYZ"]}' \
  -F 'files=@resume1.pdf' \
  -F 'files=@resume2.docx'
```

## Contributing
I welcome contributions to this project. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/your_feature_name`).
3. Make changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your_feature_name`).
5. Create a new Pull Request.
6. Please make sure to update tests as appropriate.

