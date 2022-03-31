import tkinter as tk
from tkinter import filedialog

class PathSelector(tk.Frame):
    """ Create a gui to select a path """
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        """ Create the widgets """
        self.message_label = tk.Label(self)
        self.message_label["text"] = "Please select a path"
        self.message_label.grid(row=0, column=1)

        self.path_label = tk.Label(self)
        self.path_label["text"] = "Path:"
        self.path_label.grid(row=2, column=0)

        self.path_entry = tk.Entry(self)
        self.path_entry.grid(row=2, column=1)

        self.path_button = tk.Button(self)
        self.path_button["text"] = "Browse"
        self.path_button["command"] = self.select_path
        self.path_button.grid(row=2, column=3)

        self.enter_button = tk.Button(self)
        self.enter_button["text"] = "Enter"
        self.enter_button["command"] = self.master.destroy
        self.enter_button.grid(row=2, column=4)

    def select_path(self):
        """ Select a path """
        path = filedialog.askdirectory()
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, path)
        self.path = path
