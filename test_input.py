import sys
import subprocess

# Run the interactive_chat.py with input
proc = subprocess.Popen(
    [sys.executable, "scripts/interactive_chat.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding='utf-8'
)

stdout, stderr = proc.communicate(input="你们几点开门\n", timeout=60)
print("STDOUT:")
print(stdout)
print("STDERR:")
print(stderr)
print("Return code:", proc.returncode)