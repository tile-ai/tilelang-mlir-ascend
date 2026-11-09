"""Machine metadata in a bounded Markdown header; prose is never metadata."""
import re


class Invalid(ValueError):
    pass


START = '<!-- harness-metadata -->'
END = '<!-- /harness-metadata -->'
# Older shipped templates placed fields in these named sections. Read them only
# for compatibility; new writers always emit the single bounded header.
LEGACY_SECTIONS = ('实现与验证约定', '并行成员', '实现与验证接收判定', 'Develop 退化判定',
                   '结构化验证', '结构化验收', '结构化验证身份', '标杆移植约定', '标杆移植接收')


def metadata(text):
    if START in text:
        if text.count(START) != 1 or text.count(END) != 1 or text.index(END) < text.index(START):
            raise Invalid('ambiguous metadata header')
        text = text.split(START, 1)[1].split(END, 1)[0]
        bounded = True
    else:
        bounded = False
    fields = {}
    active, fence = True, None
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(('```', '~~~')):
            marker = stripped[:3]
            fence = None if fence == marker else marker if fence is None else fence
            continue
        if fence:
            continue
        if not bounded and line.startswith('## '):
            active = any(line[3:].startswith(section) for section in LEGACY_SECTIONS)
        if not active:
            continue
        match = re.fullmatch(r'- ([A-Za-z_][A-Za-z_0-9]*): (.*)', line)
        if match:
            key, value = match.groups()
            if key in fields:
                raise Invalid('missing/duplicate ' + key)
            fields[key] = value.strip()
    return fields


def render(title, fields, body=''):
    return '# ' + title + '\n\n' + START + '\n' + ''.join(
        f'- {key}: {value}\n' for key, value in fields.items()) + END + '\n\n' + body
