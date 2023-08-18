# SupportMe - Text Classification App

SupportMe is a web-based text classification application designed to categorize support tickets into different topics. This project demonstrates the end-to-end process of building a full-stack text classification app, from data preprocessing to model inference. The model used, called the EncoderTransformer or ET for short, is a custom language model I developed using PyTorch.
<br><br>
**Note:** This repository used to contain the original model, training and data processing code. Along with a comparative research paper on the project. Since 8/18/2023 this project has been moved [here](https://github.com/chris-caballero/Ticket-Classification-Data.git).

## Features

Classify support tickets into predefined topics:
- Bank Account Services
- Credit card / Prepaid card
- Others
- Theft / Dispute reporting
- Mortgages / Loans

Choose between two different models:
- Model without Part-of-Speech (POS) tagging
- Model with POS tagging

User-friendly web interface for text input and classification

## Project Structure

- **client**: Contains the frontend code of the web application, including HTML, CSS, and JavaScript files.
- **server**: Contains the backend code for the Flask web server.
- **server/models**: Holds the trained text classification model, schema module and related utilities.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/chris-caballero/Ticket-Classification.git
   cd supportme-text-classification
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask web server:

   ```bash
   python app.py
   ```

4. Open your web browser and navigate to `http://localhost:5000` to access the SupportMe app.

## Technologies Used

- Python
- Flask
- PyTorch
- Transformers
- Bootstrap

## How It Works

1. Users input text in the web interface and select a model type.
2. The frontend sends the text and model type to the backend API.
3. The backend preprocesses the text and uses the selected model to predict the topic.
4. The predicted topic is displayed on the web interface.

## Future Enhancements

- Deploy the application to a cloud platform for wider accessibility.
- Implement user authentication and user-specific saved classifications.
- Add support for additional models and fine-tuning options.
- Improve UI/UX design and responsiveness.
- Enhance error handling and logging.
- Add synthetic FAQ database for modeling a support website functionality. This will use the model for topic embeddings and compare user request with the embedding indexed database.

## Contributing

Contributions are welcome! If you find any issues or want to add new features, feel free to submit a pull request.

