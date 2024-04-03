# Control Flow

Control Flow is a framework for integrating AI agents into traditional software. It allows for agents that can be precisely controlled, observed, and debugged, while retaining the autonomy and flexibility that make LLM agents so powerful. Control Flow agents are designed to be invoked programmatically, though they are capable of interacting with humans and other agents as well.

## Example

```python
from control_flow import flow, task, instructions
from pydantic import BaseModel

class Survey(BaseModel):
    first_name: str
    last_name: str
    age: int
    city: str
    state: str
    favorite_color: str

@task
def get_user_info() -> Survey:
    '''collect information from the user'''

@flow
def fill_out_survey():
    with instructions('talk like a pirate'):    
        survey = get_user_info()

    if survey.age < 18:
        raise ValueError('You must be 18 or older to continue.')

    return survey

fill_out_survey()
```