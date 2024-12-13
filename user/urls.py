from django.urls import path, include
from . import views

app_name = 'user'

urlpatterns = [
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('register/', views.register, name='register'),
    path('get_user_info/', views.get_user_info, name='get_user_info'),
    path('get-captcha/', views.get_captcha, name='get_captcha'),
    path('send_verification_code/', views.send_verification_code, name='send_verification_code'),
]