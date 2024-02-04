import os

def my_view(request):
    file_path = 'fpd/model.json'  # Update with the correct path
    
    if os.path.exists(file_path):
        # File exists, load the model here
        # Example code to load a JSON model file and weights (replace with your actual model loading code):
        from tensorflow import model_from_json
        
        json_file = open(file_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("fpd/model.h5")
        
        # Now you can use the loaded_model for predictions or other tasks
        
    else:
        print(f"File not found: {file_path}")
    
    # Rest of your view logic
