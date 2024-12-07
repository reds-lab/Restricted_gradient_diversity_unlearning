# utils.py
import termcolor
import os
import platform

# Determine the operating system
current_os = platform.system()

if current_os == "Windows":
    termcolor.ENABLE_WINDOWS_COLOR = True
    os.system('color')  # Only execute on Windows
else:
    termcolor.ENABLE_WINDOWS_COLOR = False
    # No need to execute 'color' on Unix-like systems

def red(content):
    return termcolor.colored(str(content), "red", attrs=["bold"])

def green(content):
    return termcolor.colored(str(content), "green", attrs=["bold"])

def blue(content):
    return termcolor.colored(str(content), "blue", attrs=["bold"])

def cyan(content):
    return termcolor.colored(str(content), "cyan", attrs=["bold"])

def yellow(content):
    return termcolor.colored(str(content), "yellow", attrs=["bold"])

def magenta(content):
    return termcolor.colored(str(content), "magenta", attrs=["bold"])

# Optional: Create a formatted section printer
def print_section(title, color="yellow"):
    color_func = globals().get(color, lambda x: x)  # Safely get the color function
    print(color_func("=" * 55))
    print(color_func(f"     {title}"))
    print(color_func("=" * 55))
