from pydantic import BaseModel, Field


class Experience(BaseModel):
    company: str = Field(description="Exclusively a company name. Never an activity name outside of work!")
    role: str = Field(description="The role when working in the company.")
    start: str = Field(description="The year when started working for the company. Always a 4-digit number.")
    end: str = Field(description="The year when finished working for the company. Always a 4-digit number or value \"present\" if still working there.")
    accomplishments: list[str] = Field(description="A list of real achievements during the work at the company.")
    used_technologies: list[str] = Field(description="Technologies used in accomplishments and responsibilites at this company.")
    responsibilities: list[str] = Field(description="A list of responsibilities during the work at the company.")


class UserProfile(BaseModel):
    professional_background: str = Field(description="Professional background of the job candidate.")
    experiences: list[Experience] = Field(description="The list of companies that the candidate worked at (exclusively companies and nothing else).")
    skills: list[str] = Field(description="Skills and abilities of the candidate.")


class JobRequirements(BaseModel):
    company_name: str = Field(description="Company name that the candidate is trying to be hired to.")
    project_description: str = Field(description="The project that the company is hiring to work on.")
    role: str = Field(description="The role the company is hiring for.")
    must_have_skills: list[str] = Field(description="The list of skills and specifically thechnologies that the company requires as a must to hire.")
    nice_to_have_skills: list[str] = Field(description="The list of skills and specifically thechnologies that the company requires as a nice to have.")
    important_experiences: list[str] = Field(description="The list of important experiences that the candidate must have experienced, f.e. working in the specified role for that past few years.")
