import os
import re
import subprocess
from resume_generator.schemas import (
    ResumeConfig,
    ArtifactsConfig,
    StrengthsSection,
    ExperienceEntry
)


tab = '    '


def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\^',
        '\\': '\\textbackslash{}',
        '<': '\\textless{}',
        '>': '\\textgreater{}',
    }
    result = ''
    for c in text:
        if c in conv:
            result += conv[c]
        else:
            result += c
    
    return result


def tex_command(name: str, tabs: int, *args: list[str]):
    args_res = ''.join([
        f'{{{arg}}}' for arg in args
    ])
    return f'{tabs * tab}\\{name}{args_res}'


def build_strengths_section(skills: list[str]):
    return (3 * tab) + ', '.join([tex_escape(skill) for skill in skills]) + '.'


def build_strengths(strengths: list[StrengthsSection]):
    result = []
    for section in strengths:
        skills = build_strengths_section(section.skills)
        result.append(tex_command(
            'CVStrengths',
            1,
            tex_escape(section.name),
            f'{3*tab}\n{skills}\n{tab}'
        ))
    return '\n\n'.join(result)


def build_experience_entry_accomplishments(accomplishments: list[str]):
    listed_things = '\n'.join([
        tex_command('CVEntryListItem', 4, tex_escape(accomplishment))
        for accomplishment in accomplishments
    ])
    
    result = tex_command('CVEntryListBegin', 3) + '\n'
    result += listed_things + "\n"
    result += tex_command('CVEntryListEnd', 3)
    
    return result


def build_experience(experience: list[ExperienceEntry]):
    result = []
    for entry in experience:
        company = entry.company
        start = entry.start
        end = entry.end
        role = entry.role
        accomplishments = build_experience_entry_accomplishments(
            entry.accomplishments
        )
        result.append(
            tex_command(
                'CVEntry',
                2,
                tex_escape(role),
                tex_escape(company),
                f'{tex_escape(start)} - {tex_escape(end)}',
                f'\n{accomplishments}\n{2 * tab}\n'
            )
        )
    return (tex_command('vspace', 2, '0.2cm') + '\n').join(result)


def generate_resume(
    resume_config: ResumeConfig,
    artifacts_conf: ArtifactsConfig,
    language: str,
    filename: str
):
    print("WORKING DIR:", os.getcwd())
    template_path = os.path.join(
        artifacts_conf.templates_path,
        f"template_{language}.tex"
    )
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    template_content = template_content.replace(
        '% about me %', tex_escape(resume_config.about_me))
    template_content = template_content.replace(
        '% strengths %', build_strengths(resume_config.strengths))
    template_content = template_content.replace(
        '% experience %', build_experience(resume_config.experience))
    
    result = subprocess.run(
        [
            "lualatex",
            f"--jobname={filename}",
            f"--output-directory={artifacts_conf.output_dir}"
        ],
        input=template_content,
        text=True,
        encoding='utf-8',
    )

    if result.returncode == 0:
        os.remove(os.path.join(artifacts_conf.output_dir, f'{filename}.aux'))
        os.remove(os.path.join(artifacts_conf.output_dir, f'{filename}.log'))
        os.remove(os.path.join(artifacts_conf.output_dir, f'{filename}.out'))
        return os.path.join(artifacts_conf.output_dir, f'{filename}.pdf')

    filepath = os.path.join(artifacts_conf.dump_dir, f'{filename}_dump.tex')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    raise SystemError("PDF generation error")
