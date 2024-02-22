from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name="index"),
    path('about', views.about, name="about"),
    path('login', views.logins, name="login"),
    path('upload', views.upload, name="upload"),
    path('register', views.register, name="register"),
    
]