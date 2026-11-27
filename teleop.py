#!/usr/bin/env python3
"""Interactive WASD & Arrow-Key Teleoperation for JetRacer MuJoCo.

Controls:
  [W] / [↑] : Throttle Forward (Accelerate)
  [S] / [↓] : Brake / Reverse
  [A] / [←] : Steer Left
  [D] / [→] : Steer Right
  [SPACE]   : Coast / Neutral (Zero Throttle & Steer)
  [B]       : Active Handbrake
  [R]       : Reset Car to Start Line
  [1]       : Switch to Autonomous Track Lap Mode
  [2]       : Switch to Manual WASD Mode
  [Q]       : Quit Teleoperation

Sends real-time UDP commands to localhost:9876.
"""

import json
import os
import select
import socket
import sys
import termios
import time
import tty

UDP_IP = "127.0.0.1"
UDP_PORT = 9876

MAX_STEER = 0.60       # rad (~34.4 deg, Waveshare JetRacer ROS /cmd_vel standard)
STEER_STEP = 0.08      # rad increment per keypress
MAX_THROTTLE = 36.92   # rad/s (= 1.2 m/s at r=0.0325m, Waveshare JetRacer rated speed)
THROTTLE_STEP = 6.0    # rad/s increment per keypress
DECAY_RATE = 0.85      # Smooth auto-centering rate when keys released


def get_key(timeout=0.04):
    """Non-blocking single key read from stdin."""
    fd = sys.stdin.fileno()
    rlist, _, _ = select.select([fd], [], [], timeout)
    if rlist:
        ch = sys.stdin.read(1)
        # Handle ANSI arrow keys: ESC [ A/B/C/D
        if ch == '\x1b':
            rlist2, _, _ = select.select([fd], [], [], 0.02)
            if rlist2:
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A': return 'w'  # Up
                    if ch3 == 'B': return 's'  # Down
                    if ch3 == 'C': return 'd'  # Right
                    if ch3 == 'D': return 'a'  # Left
        return ch
    return None


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    steer = 0.0
    throttle = 0.0
    mode = "MANUAL"
    reset_flag = False

    print("\033[2J\033[H", end="")  # Clear screen
    print("==================================================================")
    print("🏎️   JETRACER MUJOCO — REAL-TIME KEYBOARD TELEOPERATION")
    print("==================================================================")
    print("  Controls:")
    print("    [W] / [↑] : Throttle Forward    [S] / [↓] : Reverse / Brake")
    print("    [A] / [←] : Steer Left          [D] / [→] : Steer Right")
    print("    [SPACE]   : Neutral / Coast     [B]       : Handbrake")
    print("    [R]       : Reset to Start      [Q]       : Quit")
    print("    [1]       : Autonomous Lap      [2]       : Manual WASD Mode")
    print("==================================================================")
    print("  Connecting to MuJoCo Studio WebViewer (localhost:9876)...")

    try:
        tty.setraw(fd)
        last_print = time.time()

        while True:
            key = get_key(timeout=0.03)
            active_input = False

            if key:
                key_l = key.lower()
                if key_l == 'q' or key == '\x03':  # 'q' or Ctrl+C
                    break
                elif key_l == 'w':
                    throttle = min(throttle + THROTTLE_STEP, MAX_THROTTLE)
                    mode = "MANUAL"
                    active_input = True
                elif key_l == 's':
                    throttle = max(throttle - THROTTLE_STEP, -MAX_THROTTLE * 0.6)
                    mode = "MANUAL"
                    active_input = True
                elif key_l == 'a':
                    steer = min(steer + STEER_STEP, MAX_STEER)
                    mode = "MANUAL"
                    active_input = True
                elif key_l == 'd':
                    steer = max(steer - STEER_STEP, -MAX_STEER)
                    mode = "MANUAL"
                    active_input = True
                elif key == ' ':
                    throttle = 0.0
                    steer = 0.0
                    mode = "MANUAL"
                    active_input = True
                elif key_l == 'b':
                    throttle = 0.0
                    mode = "MANUAL"
                    active_input = True
                elif key_l == 'r':
                    reset_flag = True
                    throttle = 0.0
                    steer = 0.0
                    mode = "MANUAL"
                    active_input = True
                elif key == '1':
                    mode = "AUTONOMOUS_LAP"
                    throttle = 25.0
                    steer = 0.0
                    active_input = True
                elif key == '2':
                    mode = "MANUAL"
                    throttle = 0.0
                    steer = 0.0
                    active_input = True

            # Smooth decay when no key pressed in manual mode
            if not active_input and mode == "MANUAL":
                steer *= DECAY_RATE
                if abs(steer) < 0.01:
                    steer = 0.0
                throttle *= 0.96
                if abs(throttle) < 0.2:
                    throttle = 0.0

            # Send command packet to simulation
            packet = {
                "steer": round(steer, 4),
                "throttle": round(throttle, 2),
                "mode": mode,
                "reset": reset_flag
            }
            sock.sendto(json.dumps(packet).encode('utf-8'), (UDP_IP, UDP_PORT))
            reset_flag = False

            # Live ASCII HUD update (~20 Hz)
            now = time.time()
            if now - last_print > 0.05:
                speed_est = throttle * 0.0325
                steer_deg = steer * 180.0 / 3.14159
                steer_bar = "|" * int(abs(steer_deg) / 2)
                steer_dir = "◀" if steer_deg > 0 else ("▶" if steer_deg < 0 else " ")
                status_line = (
                    f"\r\033[K"
                    f"Mode: \033[1;32m{mode:14s}\033[0m | "
                    f"Steer: \033[1;33m{steer_deg:+5.1f}° {steer_dir}\033[0m | "
                    f"Throttle: \033[1;36m{throttle:+5.1f} rad/s\033[0m (~{speed_est:+.2f} m/s) "
                )
                sys.stdout.write(status_line)
                sys.stdout.flush()
                last_print = now

            time.sleep(0.01)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # Send zero-stop on exit
        try:
            stop_packet = {"steer": 0.0, "throttle": 0.0, "mode": "MANUAL", "reset": False}
            sock.sendto(json.dumps(stop_packet).encode('utf-8'), (UDP_IP, UDP_PORT))
            sock.close()
        except Exception:
            pass
        print("\n\n[Teleop] Session ended. Car safely stopped.\n")


if __name__ == '__main__':
    main()
