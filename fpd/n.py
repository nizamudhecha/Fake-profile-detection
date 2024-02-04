import numpy as np
from django.shortcuts import render
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model

# Create a simple neural network model for testing purposes
def create_simple_model(input_dim):
    model = Sequential()
    model.add(Dense(5, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create and save a simple model
input_dim = 5  # Adjust this based on the number of input features you have
simple_model = create_simple_model(input_dim)
simple_model.save("fpd/simple_model.h5")  # Save the model in HDF5 format

def Detect(request):
    if request.method == 'POST':
        try:
            # Parse input features from the POST request
            status = int(request.POST.get('status', 0))
            followers = int(request.POST.get('followers', 0))
            friends = int(request.POST.get('friends', 0))
            account_age = int(request.POST.get('account_age', 0))
            pic = int(request.POST.get('pic', 0))
            post_count = int(request.POST.get('post_count', 0))

            # Load the saved simple model
            loaded_model = load_model("fpd/simple_model.h5")

            # Prepare input features for prediction
            features = np.array([[status, followers, friends, pic,post_count]])

            # Make predictions
            prediction = loaded_model.predict(features)
            prediction = prediction[0]

            # Check the threshold and determine the result
            if prediction > 0.7:
                result = "The Profile is Fake"
            else:
                result = "The Profile is real"

            msg = result

            return render(request, 'fpd/detect.html', {'msg': msg})

        except Exception as e:
            msg = f"Error: {str(e)}"
            return render(request, 'fpd/detect.html', {'msg': msg})
    else:
        # Handle GET requests or other HTTP methods as needed
        return render(request, 'fpd/detect.html')
