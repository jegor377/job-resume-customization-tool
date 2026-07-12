from pydantic import BaseModel


class Experience(BaseModel):
    company: str
    role: str
    start: str
    end: str
    accomplishments: list[str]
    used_technologies: list[str]
    responsibilities: list[str]


class UserProfile(BaseModel):
    summary: str | None = None
    experiences: list[Experience]
    skills: list[str]


class JobRequirements(BaseModel):
    company_name: str
    project_description: str
    role: str
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    important_experiences: list[str]
