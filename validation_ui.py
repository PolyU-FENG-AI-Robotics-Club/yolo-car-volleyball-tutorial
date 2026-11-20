"""Tk desktop interface: model/video selection, side-by-side preview, and controls."""
import json
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk, ImageOps
import cv2
from detect import ROOT


class ValidationApp:
    def __init__(self, window, args):
        self.window = window
        window.title("YOLO Car Detection — Independent Video Validation")
        window.geometry("1320x850")
        window.minsize(960, 700)
        self.events = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.worker = None
        self.closing = False
        self.output = None
        self.total_frames = 0
        self.last_frame = 0
        self.references = []
        self.model = tk.StringVar(value=args.model)
        default_source = ROOT / "data/validation/traffic-720p.mp4"
        if not default_source.exists():
            default_source = ROOT / "data/validation/traffic.mp4"
        self.source = tk.StringVar(value=args.source or str(default_source))
        self.output_base = tk.StringVar(value=args.output)
        self.conf = tk.StringVar(value=str(args.conf))
        self.status = tk.StringVar(value="Ready. Select a video and model, then click Start validation.")
        self.stats = tk.StringVar(value="Frame: —    Cars in this frame: —    Inference: —")
        style = ttk.Style(window)
        style.theme_use("clam")
        style.configure("TButton", font=("Arial", 11), padding=7)
        style.configure("TLabel", font=("Arial", 11))
        style.configure("Title.TLabel", font=("Arial", 20, "bold"))
        outer = ttk.Frame(window, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Car detection · video validation", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Independent video only · CPU inference · Frames are never added to the training data").pack(anchor="w", pady=(4, 12))
        form = ttk.Frame(outer)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self.inputs = []
        for row, label, variable, mode in [(0, "Video", self.source, "video"), (1, "Model", self.model, "model"), (2, "Output folder", self.output_base, "folder")]:
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
            entry = ttk.Entry(form, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            button = ttk.Button(form, text="Browse…", command=lambda v=variable, m=mode: self.browse(v, m))
            button.grid(row=row, column=2, padx=(8, 0), pady=4)
            self.inputs.extend([entry, button])
        options = ttk.Frame(outer)
        options.pack(fill="x", pady=8)
        for text, path in [("Use pretrained model", ROOT / "models/yolo11n.pt"), ("Use trained model", ROOT / "outputs/train-only/weights/last.pt")]:
            button = ttk.Button(options, text=text, command=lambda p=path: self.model.set(str(p)))
            button.pack(side="left", padx=(0, 8));self.inputs.append(button)
        ttk.Label(options, text="Confidence threshold:").pack(side="left", padx=(12, 6))
        spin = ttk.Spinbox(options, from_=0.01, to=1, increment=0.05, textvariable=self.conf, width=6)
        spin.pack(side="left");self.inputs.append(spin)
        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 10))
        self.start_button = ttk.Button(controls, text="Start validation", command=self.start)
        self.start_button.pack(side="left", padx=(0, 8))
        self.pause_button = ttk.Button(controls, text="Pause", command=self.toggle_pause, state="disabled")
        self.pause_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(controls, text="Stop and save", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        self.play_button = ttk.Button(controls, text="Play saved video", command=self.play, state="disabled")
        self.play_button.pack(side="right")
        self.open_button = ttk.Button(controls, text="Open results", command=self.open_results, state="disabled")
        self.open_button.pack(side="right", padx=8)
        previews = ttk.Frame(outer)
        previews.pack(fill="both", expand=True)
        previews.columnconfigure(0, weight=1, uniform="preview")
        previews.columnconfigure(1, weight=1, uniform="preview")
        previews.rowconfigure(1, weight=1)
        ttk.Label(previews, text="Original video").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Label(previews, text="Detected cars · boxes and confidence").grid(row=0, column=1, sticky="w", padx=8, pady=6)
        self.left = tk.Label(previews, background="#18202b", text="Original video preview", foreground="white", font=("Arial", 15))
        self.right = tk.Label(previews, background="#18202b", text="Detection results appear here", foreground="white", font=("Arial", 15))
        self.left.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.right.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        ttk.Label(outer, textvariable=self.stats).pack(anchor="w", pady=(12, 6))
        self.progress = ttk.Progressbar(outer, maximum=100)
        self.progress.pack(fill="x")
        ttk.Label(outer, textvariable=self.status, wraplength=1240).pack(anchor="w", pady=8)
        ttk.Label(outer, text="Counts are per-frame detections, not unique vehicles. This unlabeled video does not provide accuracy or mAP.").pack(anchor="w")
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.after(100, self.poll)
        window.after(200, self.preview_source)

    def browse(self, variable, mode):
        if mode == "folder":
            value = filedialog.askdirectory(title="Choose a parent location for a new result folder")
            if value:
                variable.set(str(Path(value) / "validation-video"))
        else:
            types = [("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v")] if mode == "video" else [("YOLO model", "*.pt")]
            value = filedialog.askopenfilename(filetypes=types)
            if value:
                variable.set(value)
                if mode == "video":self.preview_source()

    def preview_source(self):
        cap = cv2.VideoCapture(self.source.get())
        ok, frame = cap.read();cap.release()
        if ok:self.show_images(frame, None)

    def show_images(self, original, annotated):
        refs = []
        for label, frame in [(self.left, original), (self.right, annotated)]:
            if frame is None:
                refs.append(getattr(label, "image", None));continue
            size = (max(label.winfo_width()-8, 200), max(label.winfo_height()-8, 140))
            im = ImageOps.contain(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), size)
            photo = ImageTk.PhotoImage(im)
            label.configure(image=photo, text="");label.image=photo;refs.append(photo)
        self.references=refs

    def start(self):
        if self.worker and self.worker.is_alive():return
        try:
            confidence=float(self.conf.get())
            if not 0<confidence<=1:raise ValueError("Confidence must be in (0, 1].")
            model=Path(self.model.get()).expanduser().resolve()
            source=Path(self.source.get()).expanduser().resolve()
            if not model.is_file():raise ValueError("The selected model does not exist.")
            if not source.is_file():raise ValueError("The selected video does not exist. Run prepare_data.py first.")
            if source.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:raise ValueError("Select a video file, not a training image.")
            cap=cv2.VideoCapture(str(source))
            if not cap.isOpened():raise ValueError("Cannot decode this video.")
            self.total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        except (ValueError, OSError) as error:
            messagebox.showerror("Cannot start validation", str(error));return
        self.stop_event.clear();self.resume_event.set();self.last_frame=0;self.progress['value']=0
        while not self.frames.empty():self.frames.get_nowait()
        self.pause_button.configure(text="Pause",state="normal");self.stop_button.configure(state="normal")
        self.start_button.configure(state="disabled");self.open_button.configure(state="disabled");self.play_button.configure(state="disabled")
        for widget in self.inputs:widget.configure(state="disabled")
        self.status.set("Loading model…")
        output=self.output_base.get()
        self.worker=threading.Thread(target=self.work,args=(model,source,output,confidence),daemon=True)
        self.worker.start()

    def work(self, model, source, output, confidence):
        try:
            from validate_video import run_validation, prepare_video
            if source == ROOT / "data/validation/traffic.mp4":
                self.events.put(('status', 'Preparing a 720p copy for desktop validation…'))
                source = prepare_video()
            def callback(result, number):
                # Keep only the newest frame so a slow display cannot build up memory.
                frame=(result.orig_img.copy(),result.plot(),number,len(result.boxes),result.speed.get('inference',0))
                try:self.frames.get_nowait()
                except queue.Empty:pass
                self.frames.put_nowait(frame)
                while not self.resume_event.wait(.1):
                    if self.stop_event.is_set():return False
                return not self.stop_event.is_set()
            saved=run_validation(model,output,source,confidence,callback,lambda s:self.events.put(('status',s)))
            self.events.put(('done',saved))
        except BaseException as error:
            self.events.put(('error',str(error)))

    def toggle_pause(self):
        if self.resume_event.is_set():
            self.resume_event.clear();self.pause_button.configure(text="Resume");self.status.set("Paused. Resume to continue or Stop and save to finish early.")
        else:
            self.resume_event.set();self.pause_button.configure(text="Pause");self.status.set("Processing video…")

    def stop(self):
        self.stop_event.set();self.resume_event.set();self.stop_button.configure(state="disabled");self.pause_button.configure(state="disabled")
        self.status.set("Stopping after the current frame; saving partial results…")

    def poll(self):
        try:
            original,annotated,n,cars,ms=self.frames.get_nowait()
            self.last_frame=n;self.show_images(original,annotated)
            self.progress['value']=100*n/max(self.total_frames,1)
            self.stats.set(f"Frame: {n} / {self.total_frames}    Cars in this frame: {cars}    Inference: {ms:.1f} ms    Device: CPU")
        except queue.Empty:pass
        while True:
            try:kind,value=self.events.get_nowait()
            except queue.Empty:break
            if kind=='status':self.status.set(value)
            else:
                self.start_button.configure(state="normal");self.pause_button.configure(state="disabled",text="Pause");self.stop_button.configure(state="disabled")
                for widget in self.inputs:widget.configure(state="normal")
                if kind=='done':
                    self.output=Path(value)
                    report=json.loads((self.output/'validation-report.json').read_text())
                    count=report['decoded_output_frames'];self.progress['value']=100*count/max(self.total_frames,1)
                    state='Complete' if report['completed'] else 'Stopped — partial results saved'
                    self.status.set(f"{state}: {count} frames. Results: {self.output}")
                    self.open_button.configure(state="normal");self.play_button.configure(state="normal")
                else:
                    self.status.set(f"Validation failed: {value}")
                    if not self.closing:messagebox.showerror("Validation failed",value)
                if self.closing:self.window.destroy();return
        self.window.after(60,self.poll)

    def open_results(self):
        if self.output:subprocess.Popen(['xdg-open',str(self.output)])

    def play(self):
        if self.output:subprocess.Popen(['xdg-open',str(self.output/'validated-traffic.mp4')])

    def close(self):
        if self.worker and self.worker.is_alive():
            self.closing=True;self.stop();self.status.set("Saving results before closing…")
        else:self.window.destroy()


def launch(args):
    try:window=tk.Tk()
    except tk.TclError as error:
        raise SystemExit(f"Cannot open the desktop display: {error}\nRun from a desktop terminal, or add --headless.")
    ValidationApp(window,args)
    window.mainloop()
