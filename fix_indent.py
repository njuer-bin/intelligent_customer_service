#!/usr/bin/env python
with open('sme_guard/rag/retrieve.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add 4 spaces of indentation before the return statement
old = 'return "\n\n".join(lines)'
new = '    return "\n\n".join(lines)'

if old in content:
    content = content.replace(old, new)
    with open('sme_guard/rag/retrieve.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Fixed indentation successfully')
else:
    print('Pattern not found in file')