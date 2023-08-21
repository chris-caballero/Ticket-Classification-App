// Get references to HTML elements
const classifyButton = document.getElementById('custom-button');
const inputText = document.getElementById('text-area-1');
const resultDiv = document.getElementById('result');

// Function to classify the text when the "Submit" button is clicked
function classifyText(event) {
    event.preventDefault();  // Prevent the default form submission

    // Get the text from the input textarea
    const text = inputText.value;

    if (text.length === 0) {
        // Display an error message if no text is provided
        document.getElementById("text-required").style.display = 'block';
        resultDiv.innerHTML = '';
        return;
    } else {
        document.getElementById("text-required").style.display = 'none';  // Hide the error message
    }

    console.log('Running classify text.');

    resultDiv.innerHTML = 'Classifying...';  // Display a message while classifying

    // Send text to backend API for classification
    fetch('/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'  // Specify JSON content type
        },
        body: JSON.stringify({ text })  // Send text as JSON
    })
    .then(function(response) {
        return response.json();
    })
    .then(function(data) {
        // Display the classification result
        resultDiv.innerHTML = `Seems like we're dealing with <strong>${data.topic}</strong>`;
        console.log(data.model);
    })
    .catch(function(error) {
        // Handle errors
        console.error('Error:', error);
        resultDiv.innerHTML = 'An error occurred while classifying.';
    });
}

// Function to handle model selection
function selectModel(button) {
    var button2;
    var model_type = button.value;
    button.classList.add("selected");

    // Deselect the other button
    if (button.value === 'pos') {
        button2 = document.getElementById('no-pos');
    } else {
        button2 = document.getElementById('pos');
    }

    if (button2.classList.contains("selected")) {
        button2.classList.remove("selected");
    }

    console.log('Choosing new model:', model_type);
    // Send a POST request to the server to select the model
    fetch('/select_model', {
        method: 'POST',
        headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: 'text=' + encodeURIComponent(model_type)
    })
    .then(function(response) {
        return response.json();
    })
    .then(function(data) {
        console.log(data.model);
    })
    .catch(function(error) {
        console.log('Error:', error);
    });
}

// Function to update character count in the textarea
function updateCharacterCount() {
    var text_length = document.getElementById("text-area-1").value.length;
    var counter = document.getElementById("word-counter");
    counter.textContent = text_length + "/" + 250 + " characters";
}

// Call the character count update function initially
updateCharacterCount();
