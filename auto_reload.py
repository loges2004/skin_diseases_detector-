# auto_reload.py
import os
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ReloadHandler(FileSystemEventHandler):
    def __init__(self, command):
        self.command = command
        self.process = None
        self.start_process()

    def start_process(self):
        if self.process:
            self.process.kill()
        self.process = subprocess.Popen(self.command, shell=True)

    def on_any_event(self, event):
        self.start_process()

if __name__ == "__main__":
    command = "waitress-serve --host=127.0.0.1 --port=8000 wsgi:app"
    event_handler = ReloadHandler(command)
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
