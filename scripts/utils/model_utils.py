import torch
import torch.nn as nn

def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def get_class_weights(y_true, num_classes, mode=1):
    total_samples = len(y_true)
    class_frequencies = torch.bincount(y_true, minlength=num_classes)
    
    if mode == 1:
        class_weights = class_frequencies / total_samples
    elif mode == 2:
        class_weights = total_samples / (len(class_frequencies) * class_frequencies)
    else:
        class_weights = None
    
    return class_weights

def train(model, dataloader, is_bert=False, class_weights=None, device=torch.device('cpu'), epochs=5, verbose=True):
    criterion = nn.CrossEntropyLoss() if class_weights is None else nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adamax(model.parameters())
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            # Forward pass
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            if is_bert:
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids)

            # Convert true labels to one-hot encoding
            labels = nn.functional.one_hot(labels, num_classes=5).type(torch.float32).to(device)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose:
            print(f'Epoch {epoch} Complete\n- Loss: {loss.item()}')
        
    return model

def evaluate(model, dataloader, is_bert=False, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            if is_bert:
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids)
            
            predictions = torch.argmax(outputs, dim=1).to(device)

            correct += (predictions == labels).sum().item()
            total += len(labels)
            
    accuracy = correct / total
         
    return accuracy

def train_and_evaluate(model=None, training_data=None, validation_data=None, is_bert=False, class_weights=None, epochs=5, verbose=True, device=torch.device('cpu')):
    '''
    CNN MODEL TRAINING
    '''
    print('Starting Training')
    model = train(model, training_data, is_bert=is_bert, class_weights=class_weights, device=device, epochs=epochs, verbose=verbose)
    '''
    CNN MODEL EVALUATION
    '''
    print('Starting Evaluation')
    train_acc = evaluate(model, training_data, is_bert=is_bert, device=device)
    test_acc = evaluate(model, validation_data, is_bert=is_bert, device=device)
    print(f'Train Accuracy: {train_acc}')
    print(f'Test Accuracy: {test_acc}')

    return model, train_acc, test_acc

def CrossValidation(model, dataset, batch_size=32, epochs=5, is_bert=False, class_weights=None, device=torch.device('cpu'), num_splits=10, verbose=True):
    from copy import deepcopy
    from sklearn.model_selection import KFold
    from torch.utils.data import DataLoader, SubsetRandomSampler

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    scores = []

    init_model = deepcopy(model)
    # Iterate over each fold
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        model = deepcopy(init_model)
        
        # Split the data into training and test sets for this fold
        train_sampler = SubsetRandomSampler(train_index)
        test_sampler = SubsetRandomSampler(test_index)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        
        # Train the model on the training set
        model = train(model, train_loader, is_bert=is_bert, class_weights=class_weights, device=device, epochs=epochs, verbose=verbose)
        
        # Evaluate the model on the test set using your evaluation function
        score = evaluate(model, test_loader, is_bert=is_bert, device=device)
        print(f"Fold {i+1} Accuracy: {score}")
        
        # Add the score for this fold to the list of scores
        scores.append(score)
    
    cross_val_score = sum(scores) / num_splits

    return scores, cross_val_score

def prepare_data(model, dataloader, config=None, is_bert=False, device=torch.device('cpu')):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        if is_bert:
            attention_mask = batch['attention_mask'].to(device)
            model.qconfig = config
            qat_transformer = torch.quantization.prepare_qat(model)
            qat_transformer(input_ids, attention_mask)
        else:
            model(input_ids)

def qat_evaluate(model, dataloader, is_bert=False, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            if is_bert:
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids)

            predictions = torch.argmax(outputs, dim=1).to(device)
            correct += (predictions == labels).sum().item()
            total += len(labels)
    
    accuracy = correct / total
         
    return accuracy


def QuantizeModel(model, config):
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        config,
        dtype=torch.qint8
    )
    return quantized_model