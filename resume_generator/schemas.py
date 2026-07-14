from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    NestedSecretsSettingsSource,
)


class StrengthsSection(BaseModel):
    name: str = Field("Skills and abilities section name.")
    skills: list[str] = Field("A list of skills of the candidate.")


class ExperienceEntry(BaseModel):
    company: str = Field(description="A company name. Never an activity name.")
    start: str = Field(description="The year when started working for the company. Always a 4-digit number.")
    end: str = Field(description="The year when finished working for the company. Always a 4-digit number or value \"present\" if still working there.")
    role: str = Field(description="The role when working in the company.")
    accomplishments: list[str] = Field(description="A list of responsibilities or real achievements during the work at the company.")


class ResumeConfig(BaseModel):
    about_me: str = Field(description="Professional background of the job candidate.")
    strengths: list[StrengthsSection] = Field(description="Skills and abilities of the candidate.")
    experience: list[ExperienceEntry] = Field(description="The list of companies that the candidate worked at (exclusively companies and nothing else).")


class ArtifactsConfig(BaseSettings):
    templates_path: str
    output_dir: str
    dump_dir: str
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_nested_delimiter='__',
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            NestedSecretsSettingsSource(file_secret_settings),
        )
