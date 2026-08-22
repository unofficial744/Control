import speech_recognition as sr
import pyautogui
import time
import re

# Added command_queue as a parameter
def listen_for_commands(command_queue=None):
    r = sr.Recognizer()
    r.energy_threshold = 300 
    r.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("\n[Voice] Active! Listening...")

        while True:
            try:
                audio = r.listen(source, timeout=None, phrase_time_limit=5)
                command = r.recognize_google(audio).lower().strip()
                print(f"[Voice] Recognized: '{command}'")

                # --- Send command to the main file's queue (for canvas mode) ---
                if command_queue is not None:
                    command_queue.put(command)

                # --- 1. Robust Close Command ---
                if "close" in command or "exit" in command:
                    pyautogui.press('esc') 
                    time.sleep(0.2)
                    pyautogui.hotkey('alt', 'f4')
                    print("[Voice Action] Forced close triggered.")

                # --- 2. Smart App Launcher (Windows Search only) ---
                elif "open" in command:
                    target = re.sub(r'\b(please|can you|kindly|open|launch)\b', '', command).strip()
                    if not target: continue

                    print(f"[Voice Action] Launching: {target}")
                    pyautogui.press('esc') 
                    time.sleep(0.2)
                    pyautogui.press('win')
                    time.sleep(0.5)
                    pyautogui.write(target, interval=0.05)
                    time.sleep(0.7)
                    pyautogui.press('enter')

                # --- 3. Smart Search & Typing (AUTO-FOCUS TEXT BOXES) ---

                # A. Auto-focus Browser Search Bar (Say: "Browse cute cats")
                elif command.startswith("browse "):
                    text_to_type = command[7:].strip()
                    if text_to_type:
                        print(f"[Voice Action] Auto-focusing URL bar and typing: '{text_to_type}'")
                        pyautogui.hotkey('ctrl', 'l') # Shortcut to jump to Browser Search Bar
                        time.sleep(0.3)
                        pyautogui.write(text_to_type, interval=0.02)
                        time.sleep(0.2)
                        pyautogui.press('enter')

                # B. Auto-focus App Search Bar (Say: "Search for specific file")
                elif command.startswith("search for "):
                    text_to_type = command[11:].strip()
                    if text_to_type:
                        print(f"[Voice Action] Auto-focusing App search and typing: '{text_to_type}'")
                        pyautogui.hotkey('ctrl', 'f') # Shortcut to jump to App Search Bar
                        time.sleep(0.3)
                        pyautogui.write(text_to_type, interval=0.02)
                        time.sleep(0.2)
                        pyautogui.press('enter')

                # C. Normal Typing without jumping (Say: "Type hello world")
                elif command.startswith("type "):
                    text_to_type = command[5:].strip()
                    if text_to_type:
                        print(f"[Voice Action] Typing out: '{text_to_type}'")
                        pyautogui.write(text_to_type, interval=0.02)

                # --- 4. Hit Enter manually ---
                elif command in ["press enter", "hit enter", "enter"]:
                    pyautogui.press('enter')
                    print("[Voice Action] Pressed 'Enter'.")

            except Exception as e:
                # pass # (Uncomment this to hide errors, but keeping it prints helpful debug info)
                pass