from django.shortcuts import render
import numpy as np
from keras.models import load_model
from instaloader import Instaloader, Profile, ProfileNotExistsException
import pandas as pd
import os
import csv
import joblib

dataset_file = 'train.csv'
model_file = 'model.pkl'  # Use the correct model file format
instagram_data_file = 'instagram_data.csv'
insta_username = 'youdoyou_123456'
insta_password = 'icandoit'
def login_to_instagram():
    L = Instaloader()
    session_file = f'.instaloader-session-{insta_username}'
    
    try:
        if os.path.exists(session_file):
            L.load_session_from_file(insta_username, filename=session_file)
        else:
            L.context.login(insta_username, insta_password)
            L.save_session_to_file(filename=session_file)
        print("Successfully logged in.")
    except Exception as e:
        print(f"Error logging in: {str(e)}")
def Index(request):
    return render(request, "fpd/detect.html")

def Detect(request):
    if request.method == 'POST':
        try:
            status = int(request.POST.get('status', 0))
            followers = int(request.POST.get('followers', 0))
            friends = int(request.POST.get('friends', 0))
            account_age = int(request.POST.get('account_age', 0))
            pic = int(request.POST.get('pic', 0))

            loaded_model = load_model("fpd/simple_model.h5")

            features = np.array([[status, followers, friends, account_age, pic]])

            prediction = loaded_model.predict(features)
            prediction = prediction[0]

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
        return render(request, 'fpd/detect.html')

def insta(request):
    login_to_instagram()
            
    
    return render(request, 'fpd/instagram.html')

def preprocess_data(profile):
    status = int(profile.mediacount)
    followers = int(profile.followers)
    friends = int(profile.followees)
    has_story = profile.has_viewable_story
    lang_num = 5

    geo = 0
    pic = 1

    features = [status, followers, friends, has_story, lang_num]

    return np.array(features)

def save_to_dataset(profile):
    if not os.path.exists(instagram_data_file):
        with open(instagram_data_file, 'w', newline='') as csvfile:
            fieldnames = ['username', 'mediacount', 'followers', 'followees', 'has_viewable_story', 'language', 'new_feature']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    username = profile.username
    mediacount = int(profile.mediacount)
    followers = int(profile.followers)
    followees = int(profile.followees)
    has_story = int(profile.has_viewable_story)
    lang_num = 5

    new_feature = 42

    with open(instagram_data_file, 'a', newline='') as csvfile:
        fieldnames = ['username', 'mediacount', 'followers', 'followees', 'has_viewable_story', 'language', 'new_feature']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'username': username, 'mediacount': mediacount, 'followers': followers, 'followees': followees, 'has_viewable_story': has_story, 'language': lang_num, 'new_feature': new_feature})
def instagram(request):
    if request.method == 'POST':
        input_username = request.POST.get('inputusername').strip()

        try:
            login_to_instagram()  # Call the login function here
            L = Instaloader()

            profile = None

            try:
                profile = Profile.from_username(L.context, input_username)
            except ProfileNotExistsException:
                msg = "The provided Instagram profile does not exist."
                return render(request, 'fpd/instagram.html', {'msg': msg})

            if profile:
                try:
                    instauserdata = preprocess_data(profile)
                    save_to_dataset(profile)
                    clf = joblib.load('model.pkl')

                    prediction = clf.predict(instauserdata.reshape(1, -1))
                    prediction = prediction[0]

                    if prediction > 0.5:
                        result = "The Profile is Fake"
                    else:
                        result = "The Profile is real"

                    msg = result

                except Exception as e:
                    msg = f"An error occurred during profile analysis: {str(e)}"
                    return render(request, 'fpd/instagram.html', {'msg': msg})

        except Exception as e:
            msg = f"An error occurred: {str(e)}"

        return render(request, 'fpd/instagram.html', {'msg': msg})

    else:
        return render(request, 'fpd/instagram.html')
