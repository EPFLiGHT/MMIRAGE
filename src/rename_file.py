import re, os

REGEX = re.compile(r'mirage', re.IGNORECASE)

def replace(m):
    x = m.group(0)

    if x == "mirage":
        return "mmirage"
    elif x == "Mirage":
        return "MMirage"
    elif x == "MIRAGE":
        return "MMIRAGE"
    else:
        raise ValueError(f"Not implemented for {x}")

for root, dirs, files in os.walk('mmirage'):
    for file in files:
        filepath = os.path.join(root, file)
        if not filepath.endswith('.py'):
            continue

        print(f'Processing {filepath}')

        # Read the file
        content = open(filepath, 'r').read()

        # Apply and replace the entire content with new content using the regex
        content = re.sub(REGEX, replace, content)

        # Finally overwrite the content
        open(filepath, 'w').write(content)


