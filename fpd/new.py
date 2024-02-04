from instaloader import Instaloader, Profile

# Initialize Instaloader
L = Instaloader()

# Specify the username of the Instagram user you want to check
username = 'beingsalmankhan'

try:
    # Retrieve the user's profile information
    profile = Profile.from_username(L.context, username)

    # Get the user's highlight reels
    highlight_reels = profile.has_highlight_reels()

    if highlight_reels:
        # Iterate through the highlight reels and print their titles
        for reel in highlight_reels:
            print(f"Highlight Reel Title: {reel.title}")

        print(f"{username} has {len(highlight_reels)} highlight reels.")
    else:
        print(f"{username} has no highlight reels.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
