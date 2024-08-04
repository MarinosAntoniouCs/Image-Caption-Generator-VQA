# Image Caption Generator and Q&A

This project is a desktop application built using PySide6 that allows users to upload an image, generate a caption for it, and ask questions about the image. The application uses pre-trained models from the Hugging Face Transformers library for image captioning and visual question answering (VQA). 

## Features

- **Image Upload**: Users can upload an image to the application.
- **Image Captioning**: The application generates a caption for the uploaded image using a pre-trained model.
- **Visual Question Answering**: Users can ask questions about the uploaded image, and the application provides answers using another pre-trained model.
- **Session Management**: Users can add new sessions, and navigate between previous and next sessions.
- **Clear Functionality**: Users can clear the uploaded image, input question, or generated answers and captions.

## User Interface

The UI is divided into several sections:
- **Top Container**: Displays the application title.
- **Middle Container**: Contains the left and right sections.
  - **Left Container**: Displays the uploaded image and generated caption.
  - **Right Container**: Allows users to input questions and displays the generated answers.
- **Bottom Container**: Contains navigation buttons for managing sessions and adding new sessions.

## Code Overview

### Main Classes and Methods

- **MainWindow**: The main class for the application window.
  - `init_ui()`: Initializes the UI components.
  - `load_models()`: Loads the pre-trained models for captioning and VQA.
  - `upload_image()`: Handles image upload functionality.
  - `clear_image()`: Clears the uploaded image.
  - `clear_all()`: Clears the input question and output answer areas.
  - `generate_caption_and_answer()`: Generates a caption and answers for the uploaded image.
  - `add_session()`, `previous_session()`, `next_session()`: Manages sessions and navigation.
  - `load_session(index)`: Loads a specific session based on the index.
  - `update_clear_image_button_state()`, `update_clear_button_state()`, `update_navigation_buttons()`: Updates the state of buttons based on the current session and inputs.

### Threads

- **CaptionModelLoader**: Loads the image captioning model in a separate thread.
- **QAModelLoader**: Loads the VQA model in a separate thread.
- **CaptionGenerator**: Generates captions for the uploaded image in a separate thread.
- **QuestionAnswerGenerator**: Generates answers for the given question about the uploaded image in a separate thread.

## Dependencies

- PySide6
- Pillow
- Transformers

## Running the Application

1. Ensure you have Python installed (preferably Python 3.8+).
2. Install the required dependencies:
    ```bash
    pip install PySide6 Pillow transformers
    ```
3. Run the application:
    ```bash
    python main.py
    ```

## Screenshots

[Include screenshots of the application interface here]

## License

[Specify the license under which the project is distributed]

## Acknowledgements

- The pre-trained models are provided by Salesforce and hosted on Hugging Face.
- UI components are built using PySide6.

## Future Improvements

- Add more advanced image processing capabilities.
- Enhance the UI/UX for better user interaction.
- Integrate more pre-trained models for diverse tasks.

## Contributing

Feel free to open issues and submit pull requests if you have any suggestions or improvements.
