import pandas as pd

data = pd.DataFrame({
    'status': [10, 20, 30, 40, 50],  # Number of posts
    'followers': [1000, 2000, 3000, 4000, 5000],
    'friends': [200, 300, 400, 500, 600],  # Followees
    'fav': [10, 15, 20, 25, 30],
    'lang_num': [5, 5, 5, 5, 5],  # Language-related feature
    'listed_count': [5, 5, 5, 5, 5],
    'geo': [0, 0, 0, 0, 0],
    'pic': [1, 1, 1, 1, 1],
    'fake': [0, 0, 1, 1, 0]  # 0 for real, 1 for fake
})
