from pydantic import BaseModel, validator

class texture_format(BaseModel):
    format: str

