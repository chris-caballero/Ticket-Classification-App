const classifyButton = document.getElementById('custom-button');
const inputText = document.getElementById('text-area-1');
const resultDiv = document.getElementById('result');

function classifyText() {
    const text = inputText.value;

    if (text.length === 0) {
        document.getElementById("text-required").style.display = 'block';
        return;
    } else {
        document.getElementById("text-required").style.display = 'none';
    }

    console.log('running classify text.');

    resultDiv.innerHTML = 'Classifying...';

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
        resultDiv.innerHTML = `Seems like were dealing with <strong>${data.topic}</strong>`;
        console.log()
    })
    .catch(function(error) {
        console.error('Error:', error);
        resultDiv.innerHTML = 'An error occurred while classifying.';
    });
}
  
function selectModel(button) {
    var button2;
    var model_type = button.value;
    button.classList.add("selected");

    if (button.value === 'pos') {
        button2 = document.getElementById('no-pos')
    } else {
        button2 = document.getElementById('pos')
    }

    if (button2.classList.contains("selected")) {
        button2.classList.remove("selected")
    }

    console.log('choosing new model:', model_type);
    // Send a POST request to the server with the text data
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
        console.log(data.model)
    })
    .catch(function(error) {
        console.log('Error:', error);
    });
}

function updateCharacterCount() {
    var text_length = document.getElementById("text-area-1").value.length;
    var counter = document.getElementById("word-counter");
    counter.textContent= text_length + "/" + 250 + " characters";
}

updateCharacterCount();