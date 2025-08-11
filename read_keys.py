import sys
import termios
import tty

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch1 = sys.stdin.read(1)
        if ch1 == '\x1b':  # ESC sequence
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                if ch3 == 'A':
                    return "UP"
                elif ch3 == 'B':
                    return "DOWN"
                elif ch3 == 'C':
                    return "RIGHT"
                elif ch3 == 'D':
                    return "LEFT"
                elif ch3 == 'F':
                    return f"END"
            else:
                return f"ESC {ch2}"
        if ch1 == '\x04':  # Ctrl+D (EOF)
            return 'CTRL_D'
        else:
            return ch1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

print("Press keys (ESC to quit):")
while True:
    key = get_key()
    if key == 'CTRL_D':
        print("CTRL_D pressed, exiting...")
        break
    print(f"Key: {key}")
