
import os
import re
import traceback


def code_only(code_string):
    """Remove # comments and redundant blank lines."""
    code_without_comments = re.sub(r"#.*", "", code_string)
    cleaned_code = re.sub(r"\n\s*\n", "\n", code_without_comments)
    return cleaned_code.strip()


def clean_code(algorithm_str):
    if isinstance(algorithm_str, str):
        return code_only(algorithm_str)
    elif isinstance(algorithm_str, list):
        return [code_only(s) for s in algorithm_str]


def extract_idea(algorithm_str):
    if isinstance(algorithm_str, str):
        return find_braces(algorithm_str)
    elif isinstance(algorithm_str, list):
        return [extract_idea(s) for s in algorithm_str]


def find_braces(response):
    """Extract content inside first {} pair."""
    match = re.search(r"\{(.*?)\}", response, re.DOTALL)
    if match:
        return match.group(1)
    if "import" in response:
        return response.split("import")[0]
    return None


def extract_code(algorithm_str):
    """Extract largest markdown code block."""
    if isinstance(algorithm_str, str):
        return find_largest_code_block_line_by_line(algorithm_str)
    elif isinstance(algorithm_str, list):
        return [extract_code(s) for s in algorithm_str]


def find_largest_code_block_line_by_line(text):
    """Find the largest ``` code block in text."""
    largest_block = ""
    current_block = ""
    nesting_level = 0
    lines = text.split("\n")

    for line in lines:
        if line.startswith("```"):
            if not line[3:].strip():  # closing delimiter
                nesting_level -= 1
                if nesting_level == 0:
                    current_block += line + "\n"
                    if len(current_block) > len(largest_block):
                        largest_block = current_block
                    current_block = ""
                else:
                    current_block += line + "\n"
            else:  # opening delimiter
                current_block += line + "\n"
                nesting_level += 1
        else:
            if nesting_level > 0:
                current_block += line + "\n"

    if largest_block:
        largest_block = "\n".join(largest_block.strip().split("\n")[1:-1])

    return largest_block if largest_block else None


def find_txt_block(string):
    """Extract content of ```txt code block."""
    inside_txt_block = False
    current_block = ""
    lines = string.split("\n")

    for line in lines:
        if line == "```txt":
            inside_txt_block = True
            current_block = line + "\n"
        elif line == "```" and inside_txt_block:
            current_block += line + "\n"
            inside_txt_block = False
        elif inside_txt_block:
            current_block += line + "\n"
    return current_block


def match_number(string):
    """Extract first integer from string, default 1."""
    match = re.search(r"\d+", string)
    if match:
        try:
            return int(match.group())
        except ValueError:
            return 1
    return 1


def read_file_as_str(path):
    with open(path, "r") as f:
        return f.read()


def write_str_to_file(s, path, mode="w"):
    if isinstance(s, list):
        s = "\n\n".join(s)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode) as f:
            f.write(s)
    except Exception as e:
        print("Failed to write to file", path, "with exception", e)
        print("Traceback:", traceback.format_exc())
        s = str(s)
        with open(path, mode) as f:
            f.write(s)
