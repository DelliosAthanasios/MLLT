import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
from translator import CodeTranslator

class TrainingGUI:
    """GUI for training the model with user data"""
    def __init__(self, translator):
        self.translator = translator
        self.root = tk.Tk()
        self.root.title("Python to C Code Translator - Training Interface")
        self.root.geometry("1200x800")
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        title_label = ttk.Label(main_frame, text="Python to C Code Translator Training", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        input_frame = ttk.LabelFrame(main_frame, text="Add Training Example", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=1)
        ttk.Label(input_frame, text="Python Code:").grid(row=0, column=0, sticky=tk.W)
        self.python_text = scrolledtext.ScrolledText(input_frame, height=8, width=50)
        self.python_text.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Label(input_frame, text="C Code:").grid(row=0, column=1, sticky=tk.W)
        self.c_text = scrolledtext.ScrolledText(input_frame, height=8, width=50)
        self.c_text.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(button_frame, text="Add Example", command=self.add_example).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear Fields", command=self.clear_fields).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Data", command=self.save_data).pack(side=tk.LEFT, padx=5)
        list_frame = ttk.LabelFrame(main_frame, text="Training Examples", padding="10")
        list_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(list_frame, columns=('Python', 'C'), show='headings', height=10)
        self.tree.heading('Python', text='Python Code')
        self.tree.heading('C', text='C Code')
        self.tree.column('Python', width=400)
        self.tree.column('C', width=400)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        train_frame = ttk.LabelFrame(main_frame, text="Training Controls", padding="10")
        train_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        param_frame = ttk.Frame(train_frame)
        param_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Label(param_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(param_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=(5, 15))
        ttk.Label(param_frame, text="Batch Size:").grid(row=0, column=2, sticky=tk.W)
        self.batch_var = tk.StringVar(value="4")
        ttk.Entry(param_frame, textvariable=self.batch_var, width=10).grid(row=0, column=3, padx=(5, 15))
        ttk.Label(param_frame, text="Learning Rate:").grid(row=0, column=4, sticky=tk.W)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(param_frame, textvariable=self.lr_var, width=10).grid(row=0, column=5, padx=(5, 0))
        train_button_frame = ttk.Frame(train_frame)
        train_button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        self.train_button = ttk.Button(train_button_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(train_button_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(train_button_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(train_button_frame, text="Open Editor", command=self.open_editor).pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(train_frame, length=400, mode='determinate')
        self.progress.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        self.status_label = ttk.Label(train_frame, text="Ready to train")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(5, 0))
        self.update_display()

    def add_example(self):
        python_code = self.python_text.get("1.0", tk.END).strip()
        c_code = self.c_text.get("1.0", tk.END).strip()
        if not python_code or not c_code:
            messagebox.showwarning("Warning", "Please enter both Python and C code.")
            return
        self.translator.add_training_example(python_code, c_code)
        self.update_display()
        self.clear_fields()
        messagebox.showinfo("Success", "Training example added!")

    def clear_fields(self):
        self.python_text.delete("1.0", tk.END)
        self.c_text.delete("1.0", tk.END)

    def update_display(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for py_code, c_code in self.translator.training_data:
            py_display = py_code[:100] + "..." if len(py_code) > 100 else py_code
            c_display = c_code[:100] + "..." if len(c_code) > 100 else c_code
            self.tree.insert("", tk.END, values=(py_display, c_display))
        count = len(self.translator.training_data)
        self.status_label.config(text=f"Training examples: {count}")

    def load_data(self):
        filepath = filedialog.askopenfilename(
            title="Load Training Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            if self.translator.load_training_data(filepath):
                self.update_display()
                messagebox.showinfo("Success", f"Loaded {len(self.translator.training_data)} training examples.")
            else:
                messagebox.showerror("Error", "Failed to load training data.")

    def save_data(self):
        if not self.translator.training_data:
            messagebox.showwarning("Warning", "No training data to save.")
            return
        filepath = filedialog.asksaveasfilename(
            title="Save Training Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.translator.save_training_data(filepath)
            messagebox.showinfo("Success", "Training data saved successfully.")

    def start_training(self):
        if not self.translator.training_data:
            messagebox.showwarning("Warning", "No training data available.")
            return
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_var.get())
            learning_rate = float(self.lr_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid training parameters.")
            return
        self.train_button.config(state='disabled')
        self.progress['value'] = 0
        self.status_label.config(text="Training in progress...")
        def training_thread():
            try:
                self.translator.train(
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    progress_callback=self.update_progress
                )
                self.root.after(0, self.training_complete)
            except Exception as e:
                self.root.after(0, lambda: self.training_error(str(e)))
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()

    def update_progress(self, current_epoch, total_epochs, loss):
        progress = (current_epoch / total_epochs) * 100
        self.root.after(0, lambda: self.progress.config(value=progress))
        self.root.after(0, lambda: self.status_label.config(text=f"Epoch {current_epoch}/{total_epochs}, Loss: {loss:.4f}"))

    def training_complete(self):
        self.train_button.config(state='normal')
        self.progress['value'] = 100
        self.status_label.config(text="Training completed successfully!")
        messagebox.showinfo("Success", "Model training completed!")

    def training_error(self, error_msg):
        self.train_button.config(state='normal')
        self.progress['value'] = 0
        self.status_label.config(text="Training failed.")
        messagebox.showerror("Error", f"Training failed: {error_msg}")

    def save_model(self):
        if self.translator.model is None:
            messagebox.showwarning("Warning", "No trained model to save.")
            return
        filepath = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pth",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.translator.save_model(filepath)
                messagebox.showinfo("Success", "Model saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    def load_model(self):
        filepath = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.translator.load_model(filepath)
                messagebox.showinfo("Success", "Model loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def open_editor(self):
        if self.translator.model is None:
            messagebox.showwarning("Warning", "Please train or load a model first.")
            return
        EditorGUI(self.translator)

    def run(self):
        self.root.mainloop()

class EditorGUI:
    """GUI for testing translations"""
    def __init__(self, translator):
        self.translator = translator
        self.root = tk.Toplevel()
        self.root.title("Python to C Code Translator - Editor")
        self.root.geometry("1000x600")
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        title_label = ttk.Label(main_frame, text="Python to C Code Translator", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        python_frame = ttk.LabelFrame(main_frame, text="Python Code", padding="10")
        python_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        python_frame.columnconfigure(0, weight=1)
        python_frame.rowconfigure(0, weight=1)
        self.python_input = scrolledtext.ScrolledText(python_frame, height=20, width=40)
        self.python_input.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        c_frame = ttk.LabelFrame(main_frame, text="C Code Output", padding="10")
        c_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        c_frame.columnconfigure(0, weight=1)
        c_frame.rowconfigure(0, weight=1)
        self.c_output = scrolledtext.ScrolledText(c_frame, height=20, width=40, state='disabled')
        self.c_output.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(20, 0))
        self.translate_button = ttk.Button(
            button_frame, 
            text="Translate Python to C", 
            command=self.translate,
            width=20
        )
        self.translate_button.pack()
        self.status_label = ttk.Label(button_frame, text="Ready to translate")
        self.status_label.pack(pady=(10, 0))
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def translate(self):
        python_code = self.python_input.get("1.0", tk.END).strip()
        if not python_code:
            self.status_label.config(text="Please enter Python code to translate")
            return
        self.status_label.config(text="Translating...")
        self.translate_button.config(state='disabled')
        try:
            c_code = self.translator.translate(python_code)
            self.c_output.config(state='normal')
            self.c_output.delete("1.0", tk.END)
            self.c_output.insert(tk.END, c_code)
            self.c_output.config(state='disabled')
            self.status_label.config(text="Translation complete!")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            messagebox.showerror("Translation Error", f"Failed to translate code: {str(e)}")
        finally:
            self.translate_button.config(state='normal') 