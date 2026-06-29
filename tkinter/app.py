import tkinter as tk
import logging
from src.predict import predict_drawing

logging.basicConfig(level=logging.INFO)



canvas_width = 256
canvas_height = 256

current_stroke_x = []
current_stroke_y = []
current_stroke_t = []
all_strokes = []

last_drag_x = None
last_drag_y = None



def on_click(event):
    """
        Signals when the user presses on the mouse to begin new drawing.
    """

    global last_drag_x, last_drag_y, start_time_epoch
    global current_stroke_x, current_stroke_y, current_stroke_t

    last_drag_x = event.x
    last_drag_y = event.y

    current_stroke_x = [event.x]
    current_stroke_y = [event.y]

    current_stroke_t = []

def on_drag(event):
    """
        Signals when the user drags the mouse while holding down the left button.
    """

    global last_drag_x, last_drag_y

    if 0 <= event.x < canvas_width and 0 <= event.y < canvas_height:
        current_stroke_x.append(event.x)
        current_stroke_y.append(event.y)
        
        canvas.create_line(
            last_drag_x, last_drag_y, event.x, event.y, 
            fill="black", width=5, capstyle=tk.ROUND, smooth=True
        )
        
        last_drag_x = event.x
        last_drag_y = event.y

def on_release(event):
    """
        Signals when the user releases the left button, convert the canvas into processed data for shape prediction.
    """

    global all_strokes, current_stroke_t
    
    if len(current_stroke_x) > 1:
        current_stroke_t = [(i + 1) * 10 for i in range(len(current_stroke_x))]

        finished_stroke = [current_stroke_x, current_stroke_y, current_stroke_t]
        all_strokes.append(finished_stroke)
        logging.info(f"Stroke locked. Points collected: {len(current_stroke_x)}. Total strokes on canvas: {len(all_strokes)}")



def execute_ui_prediction():
    """
        Forwards tracked drawing to the model for shape prediction.
    """

    logging.info("Predicting drawn shape...")
    
    predicted_label, confidence = predict_drawing(all_strokes)
    
    result_text_var.set(f"Prediction: {predicted_label.capitalize()}")
    confidence_text_var.set(f"Certainty: {confidence:.2f}%")
    
    print(f"Terminal Log -> Guess: {predicted_label} | Accuracy Probability: {confidence:.4f}%")

def execute_canvas_clear():
    """
        Reset canvas and all stored variables.
    """

    global all_strokes, current_stroke_x, current_stroke_y, current_stroke_t

    canvas.delete("all")
    all_strokes = []
    current_stroke_x = []
    current_stroke_y = []
    current_stroke_t = []
    last_drag_x = None
    last_drag_y = None

    result_text_var.set("Prediction: None")
    confidence_text_var.set("Certainty: 0.00%")
    logging.info("Canvas reset.")




root = tk.Tk()
root.title("Pydraw Neural Network Shape Classifier")

canvas = tk.Canvas(
    root, 
    width = canvas_width, height = canvas_height, 
    bg = 'white', 
    highlightthickness=1, 
    highlightbackground="gray")
canvas.pack(pady=10)

canvas.bind("<Button-1>", on_click)
canvas.bind("<B1-Motion>", on_drag)
canvas.bind("<ButtonRelease-1>", on_release)

result_text_var = tk.StringVar(value="Prediction: None")
confidence_text_var = tk.StringVar(value="Certainty: 0.00%")

label_result = tk.Label(root, textvariable=result_text_var, font=("Helvetica", 12, "bold"))
label_result.pack()
label_confidence = tk.Label(root, textvariable=confidence_text_var, font=("Helvetica", 10))
label_confidence.pack(pady=2)



button_frame = tk.Frame(root)
button_frame.pack(pady=10)

predict_button = tk.Button(button_frame, text="Predict Shape", command=execute_ui_prediction, width=12, bg="#4CAF50", fg="white")
predict_button.grid(row=0, column=0, padx=5)

erase_button = tk.Button(button_frame, text="Clear Screen", command=execute_canvas_clear, width=12, bg="#f44336", fg="white")
erase_button.grid(row=0, column=1, padx=5)

root.mainloop()