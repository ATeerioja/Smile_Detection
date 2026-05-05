import subprocess
import sys
import os
import tkinter as tk

#Funktio, joka avaa scriptin
def run_script(script_name):
    subprocess.Popen([sys.executable, script_name])

#Luo yksinkertaisen käyttöliittymän
root = tk.Tk()
root.title("Smile Detection Launcher")
root.geometry("300x150")

label = tk.Label(root, text="Choose mode:", font=("Arial", 14))
label.pack(pady=10)

btn_main = tk.Button(root, text="Run Main",
                     command=lambda: run_script("face_detection.py"),
                     width=20)
btn_main.pack(pady=5)

btn_debug = tk.Button(root, text="Run Debug",
                      command=lambda: run_script("face_detection_debug.py"),
                      width=20)
btn_debug.pack(pady=5)

root.mainloop()