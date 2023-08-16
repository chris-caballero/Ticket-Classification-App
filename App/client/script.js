const classifyButton = document.getElementById('classifyButton');
const inputText = document.getElementById('inputText');
const resultDiv = document.getElementById('result');

classifyButton.addEventListener('click', async () => {
  const text = inputText.value.trim();

  if (text === '') {
      resultDiv.textContent = 'Please enter text.';
      return;
  }

  resultDiv.textContent = 'Classifying...';

  // Send text to backend API for classification
  try {
      const response = await fetch('/classify', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'  // Specify JSON content type
          },
          body: JSON.stringify({ text })  // Send text as JSON
      });

      if (response.ok) {
          const data = await response.json();  // Parse JSON response
          resultDiv.textContent = `Predicted Class: ${data.predicted_class}\n\nIt seems like you have a ${data.topic} problem`;  // Access 'predicted_class' key
      } else {
          resultDiv.textContent = 'Failed to classify.';
      }
  } catch (error) {
      console.error('Error:', error);
      resultDiv.textContent = 'An error occurred while classifying.';
  }
});