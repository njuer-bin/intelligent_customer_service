#!/usr/bin/env python
import sys

with open('sme_guard/rag/retrieve.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The problem is on lines with multi-line f-strings
# We need to replace the malformed f-string

# Pattern: lines with f"{header}\n{content}" inside a list.append
# Let's find and fix it

# Look for the specific malformed pattern
# The issue is: f"{header}\n{content}" inside a list.append()
# We need to change it to: f"{header} {content}"

# Let's try a different approach - just rewrite the function entirely
# by finding 'def format_retrieval_results' and replacing everything until the end of the function

# Find the function start
start_idx = content.find('def format_retrieval_results(')
if start_idx < 0:
    print('ERROR: Could not find format_retrieval_results function')
    sys.exit(1)

# Find the end of the function (next function or end of file)
# Look for the next top-level def or the end
rest = content[start_idx:]

# Simple approach: replace the known problematic section
# The problematic code is:
# lines.append(
#     f"{header}\n{content}"
# )

# Replace with:
# lines.append(
#     f"{header} {content}"
# )

old_code = """        lines.append(
            f"{header}
{content}"
        )"""

new_code = """        lines.append(
            f"{header} {content}"
        )"""

if old_code in content:
    content = content.replace(old_code, new_code)
    print('Successfully replaced the malformed f-string')
else:
    print('Could not find the exact pattern to replace')
    # Try alternative - search for the return statement pattern
    if 'return "\\n\\n".join(lines)' in content:
        print('Found the return statement pattern')
        # Just verify the file can be parsed
        try:
            compile(content, 'retrieve.py', 'exec')
            print('File compiles successfully')
        except SyntaxError as e:
            print('Syntax error:', e)
    else:
        print('Alternative pattern not found either')

with open('sme_guard/rag/retrieve.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('File written successfully')