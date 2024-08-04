BUTTON_STYLE_NORMAL = """
    background-color: #6a0dad; 
    color: white; 
    border: 2px solid #555; 
    border-radius: 20px; 
    padding: 5px; 
    font-family: 'Arial'; 
    font-size: 14px; 
    font-weight: bold;
"""

BUTTON_STYLE_PRESSED = """
    background-color: #4a0072; 
    color: white; 
    border: 2px solid #555; 
    border-radius: 20px; 
    padding: 5px; 
    font-family: 'Arial'; 
    font-size: 14px; 
    font-weight: bold;
"""

DISABLED_STYLE = """
    background-color: #9b6fd3; 
    color: #d3d3d3; 
    border: 2px solid #555; 
    border-radius: 20px; 
    padding: 5px; 
    font-family: 'Arial'; 
    font-size: 14px; 
    font-weight: bold;
"""

STATUS_STYLE = """
    background-color: #f0f0f0;
    color: grey;
    font-style: italic;
"""

import sys
import requests
from PIL import Image
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QLabel, QTextEdit, QPushButton, QWidget, QFrame, QFileDialog, 
    QSizePolicy
)
from PySide6.QtGui import QFont, QPixmap, QIcon
from PySide6.QtCore import Qt, QThread, Signal
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

class CaptionModelLoader(QThread):
    model_loaded = Signal(object, object)

    def run(self):
        caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model_loaded.emit(caption_processor, caption_model)

class QAModelLoader(QThread):
    model_loaded = Signal(object, object)

    def run(self):
        qa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        qa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        self.model_loaded.emit(qa_processor, qa_model)

class CaptionGenerator(QThread):
    caption_generated = Signal(str)

    def __init__(self, processor, model, image_path):
        super().__init__()
        self.processor = processor
        self.model = model
        self.image_path = image_path

    def run(self):
        raw_image = Image.open(self.image_path).convert('RGB')
        caption_inputs = self.processor(raw_image, return_tensors="pt")
        caption_outputs = self.model.generate(**caption_inputs, max_new_tokens=50)
        caption = self.processor.decode(caption_outputs[0], skip_special_tokens=True)
        self.caption_generated.emit(caption)
        
class QuestionAnswerGenerator(QThread):
    answer_generated = Signal(str)

    def __init__(self, processor, model, image_path, question):
        super().__init__()
        self.processor = processor
        self.model = model
        self.image_path = image_path
        self.question = question

    def run(self):
        raw_image = Image.open(self.image_path).convert('RGB')
        qa_inputs = self.processor(raw_image, self.question, return_tensors="pt")
        qa_outputs = self.model.generate(**qa_inputs, max_new_tokens=50)
        answer = self.processor.decode(qa_outputs[0], skip_special_tokens=True)
        self.answer_generated.emit(answer)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Caption Generator and Q&A")
        self.setWindowIcon(QIcon("picture.png"))  # Set the window icon with a relative path

        # Initialize UI elements
        self.init_ui()

        # Placeholder for models
        self.caption_processor = None
        self.caption_model = None
        self.qa_processor = None
        self.qa_model = None

        # Flags to track model loading
        self.caption_model_loaded = False
        self.qa_model_loaded = False

        # Load models using threads
        self.load_models()

    def init_ui(self):
        # Initialize session data
        self.sessions = []
        self.current_session_index = 0  # Start with the first session index

        self.image_uploaded = False  # Track whether an image is uploaded
        self.text_in_input_area = False  # Track whether there is text in the input area

        # Initialize the first session
        self.sessions.append({
            'image': None,
            'caption': '',
            'question': '',
            'answer': ''
        })

        # Create the main layout
        main_layout = QVBoxLayout()

        # Create the top container for the title
        top_container = QFrame()
        top_container.setFrameShape(QFrame.Box)
        top_container.setFixedHeight(100)  # Increase height for symmetry
        top_layout = QVBoxLayout(top_container)
        top_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        title_label = QLabel("Image Caption Generator and Q&A")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Times New Roman", 34, QFont.Bold))
        title_label.setStyleSheet("color: white;")  # White text color
        top_layout.addWidget(title_label)

        # Create the middle container
        middle_container = QFrame()
        middle_container.setFrameShape(QFrame.Box)
        middle_layout = QHBoxLayout(middle_container)
        middle_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        # Create the left container for image and caption
        left_container = QFrame()
        left_container.setFrameShape(QFrame.Box)
        left_layout = QVBoxLayout(left_container)
        left_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        # Split the left container horizontally
        upper_left_container = QFrame()
        upper_left_layout = QVBoxLayout(upper_left_container)
        upper_left_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)  # Connect button to upload_image method
        self.upload_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.upload_button.setFixedSize(150, 40)  # Set fixed size for the button
        self.upload_button.setStyleSheet(DISABLED_STYLE)
        self.upload_button.setEnabled(False)  # Initialize as disabled
        self.upload_button.pressed.connect(lambda: self.upload_button.setStyleSheet(BUTTON_STYLE_PRESSED))
        self.upload_button.released.connect(lambda: self.upload_button.setStyleSheet(BUTTON_STYLE_NORMAL))

        self.clear_image_button = QPushButton("Clear Image")
        self.clear_image_button.clicked.connect(self.clear_image)  # Connect button to clear_image method
        self.clear_image_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.clear_image_button.setFixedSize(150, 40)  # Set fixed size for the button
        self.clear_image_button.setEnabled(False)  # Disable the button initially
        self.clear_image_button.setStyleSheet(DISABLED_STYLE)
        self.clear_image_button.pressed.connect(lambda: self.clear_image_button.setStyleSheet(BUTTON_STYLE_PRESSED))
        self.clear_image_button.released.connect(lambda: self.clear_image_button.setStyleSheet(BUTTON_STYLE_NORMAL))

        # Create a horizontal layout to center the buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.clear_image_button)
        button_layout.addStretch()

        upper_left_layout.addWidget(self.image_label)
        upper_left_layout.addLayout(button_layout)

        lower_left_container = QFrame()
        lower_left_layout = QVBoxLayout(lower_left_container)
        lower_left_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        self.caption_area = QTextEdit()
        self.caption_area.setPlaceholderText("Caption generated:")
        self.caption_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Always show vertical scroll bar
        self.caption_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show horizontal scroll bar as needed
        self.caption_area.setStyleSheet(
            "border: 2px solid #6a0dad; border-radius: 5px; padding: 5px; "
            "background-color: #f0f0f0; color: black;"
        )  # Purple border, light gray background, black text for caption area
        lower_left_layout.addWidget(self.caption_area)

        left_layout.addWidget(upper_left_container)
        left_layout.addWidget(lower_left_container)

        # Ensure equal space distribution
        left_layout.setStretchFactor(upper_left_container, 1)
        left_layout.setStretchFactor(lower_left_container, 1)

        # Create the right container for input and output areas
        right_container = QFrame()
        right_container.setFrameShape(QFrame.Box)
        right_layout = QVBoxLayout(right_container)
        right_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        self.input_area = QTextEdit()
        self.input_area.setPlaceholderText("Ask a question:")
        self.input_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Always show vertical scroll bar
        self.input_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show horizontal scroll bar as needed
        self.input_area.setStyleSheet(
            "border: 2px solid #6a0dad; border-radius: 5px; padding: 5px; "
            "background-color: #f0f0f0; color: black;"
        )  # Purple border, light gray background, black text for input area
        self.input_area.textChanged.connect(self.update_clear_button_state)  # Connect textChanged signal

        right_layout.addWidget(self.input_area)

        # Create the generate and clear buttons
        self.generate_button = QPushButton("Generate")
        self.generate_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.generate_button.setFixedSize(150, 40)  # Set fixed size for the button
        self.generate_button.clicked.connect(self.generate_caption_and_answer)  # Connect to generate_caption_and_answer method
        self.generate_button.setStyleSheet(DISABLED_STYLE)
        self.generate_button.setEnabled(False)
        self.generate_button.pressed.connect(lambda: self.generate_button.setStyleSheet(BUTTON_STYLE_PRESSED))
        self.generate_button.released.connect(lambda: self.generate_button.setStyleSheet(BUTTON_STYLE_NORMAL))

        self.clear_button = QPushButton("Clear")
        self.clear_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.clear_button.setFixedSize(150, 40)  # Set fixed size for the button
        self.clear_button.clicked.connect(self.clear_all)  # Connect to clear_all method
        self.update_clear_button_state()  # Set initial state of the button
        self.clear_button.setStyleSheet(DISABLED_STYLE)
        self.clear_button.pressed.connect(lambda: self.clear_button.setStyleSheet(BUTTON_STYLE_PRESSED))
        self.clear_button.released.connect(lambda: self.clear_button.setStyleSheet(BUTTON_STYLE_NORMAL))

        # Create a horizontal layout to center the buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        right_layout.addLayout(button_layout)

        self.output_area = QTextEdit()
        self.output_area.setPlaceholderText("Answer:")
        self.output_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Always show vertical scroll bar
        self.output_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show horizontal scroll bar as needed
        self.output_area.setStyleSheet(
            "border: 2px solid #6a0dad; border-radius: 5px; padding: 5px; "
            "background-color: #f0f0f0; color: black;"
        )  # Purple border, light gray background, black text for output area
        right_layout.addWidget(self.output_area)

        middle_layout.addWidget(left_container)
        middle_layout.addWidget(right_container)

        # Create the bottom container for buttons
        bottom_container = QFrame()
        bottom_container.setFrameShape(QFrame.Box)
        bottom_container.setFixedHeight(80)  # Increase height for symmetry
        bottom_layout = QHBoxLayout(bottom_container)
        bottom_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        self.add_button = QPushButton("Add session")
        self.prev_button = QPushButton("Previous session")
        self.next_button = QPushButton("Next session")

        # Apply purple background and white text to bottom buttons
        for button in [self.prev_button, self.next_button, self.add_button]:
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.setFixedSize(150, 40)  # Set fixed size for the button
            button.setStyleSheet(BUTTON_STYLE_NORMAL)
            button.pressed.connect(lambda b=button: b.setStyleSheet(BUTTON_STYLE_PRESSED))
            button.released.connect(lambda b=button: b.setStyleSheet(BUTTON_STYLE_NORMAL))

        self.add_button.clicked.connect(self.add_session)
        self.prev_button.clicked.connect(self.previous_session)
        self.next_button.clicked.connect(self.next_session)

        # Initialize buttons as disabled if no sessions
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.prev_button.setStyleSheet(DISABLED_STYLE)
        self.next_button.setStyleSheet(DISABLED_STYLE)

        bottom_layout.addWidget(self.add_button)
        bottom_layout.addWidget(self.prev_button)
        bottom_layout.addWidget(self.next_button)

        # Add all containers to the main layout
        main_layout.addWidget(top_container)
        main_layout.addWidget(middle_container)
        main_layout.addWidget(bottom_container)

        # Set the central widget and layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Load the initial session
        self.load_session(self.current_session_index)
        self.update_navigation_buttons()


    def load_models(self):
        # Load caption model
        self.caption_loader = CaptionModelLoader()
        self.caption_loader.model_loaded.connect(self.on_caption_model_loaded)
        self.caption_loader.start()

        # Load QA model
        self.qa_loader = QAModelLoader()
        self.qa_loader.model_loaded.connect(self.on_qa_model_loaded)
        self.qa_loader.start()

    def on_caption_model_loaded(self, processor, model):
        self.caption_processor = processor
        self.caption_model = model
        self.caption_model_loaded = True
        print("Caption model loaded")
    
        # Enable the upload button when the caption model is loaded
        self.upload_button.setEnabled(True)
        self.upload_button.setStyleSheet(BUTTON_STYLE_NORMAL)


    def on_qa_model_loaded(self, processor, model):
        self.qa_processor = processor
        self.qa_model = model
        self.qa_model_loaded = True
        print("QA model loaded")

    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Upload Image", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif)")
    
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.current_image_path = file_path
            self.image_uploaded = True
            self.update_clear_image_button_state()

            # Disable the generate button and show status message in caption area
            self.generate_button.setEnabled(False)
            self.generate_button.setStyleSheet(DISABLED_STYLE)
            self.caption_area.setText("Caption is being generated...")
            self.caption_area.setStyleSheet(STATUS_STYLE)

            # Clear the output area when a new image is uploaded
            self.output_area.clear()
            self.output_area.setStyleSheet("background-color: #f0f0f0; color: black;")  # Set to default white background

            # Check if the caption model is loaded before starting the caption generation thread
            if self.caption_model_loaded:
                self.generate_caption_thread = CaptionGenerator(self.caption_processor, self.caption_model, file_path)
                self.generate_caption_thread.caption_generated.connect(self.on_caption_generated)
                self.generate_caption_thread.start()
            else:
                self.output_area.setText("Caption model is still loading, please wait...")
                self.output_area.setStyleSheet(STATUS_STYLE)

    def on_caption_generated(self, caption):
        self.caption_area.setText(caption)
        self.caption_area.setStyleSheet("background-color: #f0f0f0; color: black;")  # Reset to default style
        self.generate_button.setEnabled(True)
        self.generate_button.setStyleSheet(BUTTON_STYLE_NORMAL)

        # Reset the output area style to default
        self.output_area.clear()
        self.output_area.setStyleSheet("background-color: #f0f0f0; color: black;")  # Reset to default white background

    def generate_caption_and_answer(self):
        if not self.image_uploaded:
            self.output_area.setText("Please upload an image first.")
            self.output_area.setStyleSheet("background-color: #f0f0f0; color: black;")  # Reset to default style
            return

        if not self.caption_processor or not self.caption_model or not self.qa_processor or not self.qa_model:
            self.output_area.setText("Models are still loading, please wait.")
            self.output_area.setStyleSheet("background-color: #f0f0f0; color: black;")  # Reset to default style
            return

        # Show status message for answer generation
        self.output_area.setText("Answer is being generated...")
        self.output_area.setStyleSheet(STATUS_STYLE)

        # Get the question from input area
        question = self.input_area.toPlainText().strip()
        if question:
            # Start the QA generation thread
            self.qa_thread = QuestionAnswerGenerator(self.qa_processor, self.qa_model, self.current_image_path, question)
            self.qa_thread.answer_generated.connect(self.on_answer_generated)
            self.qa_thread.start()
        else:
            self.output_area.setText("")
            self.output_area.setStyleSheet("background-color: #f0f0f0; color: black;")  # Reset to default style

    def on_answer_generated(self, answer):
        self.output_area.setText(answer)
        self.output_area.setStyleSheet("background-color: #f0f0f0; color: black;")  # Reset to default style


    def clear_image(self):
        # Clear the image label and caption area
        self.image_label.clear()
        self.caption_area.clear()
        self.image_uploaded = False
        self.update_clear_image_button_state()

    def clear_all(self):
        # Clear input and output areas only
        self.input_area.clear()
        self.output_area.clear()
        self.update_clear_button_state()

    def add_session(self):
        print("Adding session...")
        print(f"Current session index before adding: {self.current_session_index}")

        # Save the current session before adding a new one
        self.sessions[self.current_session_index] = {
            'image': self.image_label.pixmap(),
            'caption': self.caption_area.toPlainText(),
            'question': self.input_area.toPlainText(),
            'answer': self.output_area.toPlainText()
        }
        print(f"Session {self.current_session_index} saved.")

        # Add new session and set it as the current session
        self.sessions.append({
            'image': None,
            'caption': '',
            'question': '',
            'answer': ''
        })
        self.current_session_index += 1  # Increment index to point to the new session
        print(f"New session added. Current session index: {self.current_session_index}")

        # Load the new session
        self.load_session(self.current_session_index)
        self.update_navigation_buttons()

        print(f"Sessions: {self.sessions}")
        print(f"Current session index after adding: {self.current_session_index}")

    def previous_session(self):
        if self.current_session_index > 0:
            # Save current session before switching
            self.sessions[self.current_session_index] = {
                'image': self.image_label.pixmap(),
                'caption': self.caption_area.toPlainText(),
                'question': self.input_area.toPlainText(),
                'answer': self.output_area.toPlainText()
            }
            self.current_session_index -= 1
            self.load_session(self.current_session_index)
            self.update_navigation_buttons()

    def next_session(self):
        if self.current_session_index < len(self.sessions) - 1:
            # Save current session before switching
            self.sessions[self.current_session_index] = {
                'image': self.image_label.pixmap(),
                'caption': self.caption_area.toPlainText(),
                'question': self.input_area.toPlainText(),
                'answer': self.output_area.toPlainText()
            }
            self.current_session_index += 1
            self.load_session(self.current_session_index)
            self.update_navigation_buttons()

    def load_session(self, index):
        session = self.sessions[index]
        if session['image']:
            self.image_label.setPixmap(session['image'])
        else:
            self.image_label.clear()
        self.caption_area.setText(session['caption'])
        self.input_area.setText(session['question'])
        self.output_area.setText(session['answer'])
        self.image_uploaded = session['image'] is not None
        self.update_clear_image_button_state()
        self.update_clear_button_state()

    def update_clear_image_button_state(self):
        if self.image_label.pixmap() and not self.image_label.pixmap().isNull():
            self.clear_image_button.setEnabled(True)
            self.clear_image_button.setStyleSheet(BUTTON_STYLE_NORMAL)
        else:
            self.clear_image_button.setEnabled(False)
            self.clear_image_button.setStyleSheet(DISABLED_STYLE)

    def update_clear_button_state(self):
        if self.input_area.toPlainText().strip():  # Check if there is text in the input area
            self.clear_button.setEnabled(True)
            self.clear_button.setStyleSheet(BUTTON_STYLE_NORMAL)
        else:
            self.clear_button.setEnabled(False)
            self.clear_button.setStyleSheet(DISABLED_STYLE)

    def update_navigation_buttons(self):
        # Enable the previous button if there is a previous session
        if self.current_session_index > 0:
            self.prev_button.setEnabled(True)
            self.prev_button.setStyleSheet(BUTTON_STYLE_NORMAL)
        else:
            self.prev_button.setEnabled(False)
            self.prev_button.setStyleSheet(DISABLED_STYLE)

        # Enable the next button if there is a next session
        if self.current_session_index < len(self.sessions) - 1:
            self.next_button.setEnabled(True)
            self.next_button.setStyleSheet(BUTTON_STYLE_NORMAL)
        else:
            self.next_button.setEnabled(False)
            self.next_button.setStyleSheet(DISABLED_STYLE)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)  # Adjust window size as needed
    window.show()
    sys.exit(app.exec())

