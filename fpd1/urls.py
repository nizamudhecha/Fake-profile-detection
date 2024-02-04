from django.contrib import admin
from django.urls import path, include
from . import views  # Import views from the fpd1 app
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.Index, name="index"),
    path('index', views.Index, name="index"),
    path("detect/", views.Detect, name="detect"),
   # path('tweet/', views.tweet, name='tweet'),
    #path("twitter/", views.twitter, name="twitter"),
    path('insta/', views.insta, name='insta'),
    path("instagram/", views.instagram, name="instagram"),
]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
