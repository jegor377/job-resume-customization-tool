from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    NestedSecretsSettingsSource,
)


class StrengthsSection(BaseModel):
    name: str
    skills: list[str]


class ExperienceEntry(BaseModel):
    company: str
    start: str
    end: str
    role: str
    accomplishments: list[str]


class ResumeConfig(BaseModel):
    about_me: str
    strengths: list[StrengthsSection]
    experience: list[ExperienceEntry]


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
