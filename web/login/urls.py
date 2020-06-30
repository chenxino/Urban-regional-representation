from django.urls import path
from . import views

app_name = 'login'
urlpatterns = [
    path('', views.index, name='index'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('logout', views.logout, name='logout'),
    path('xx',views.xx,name ='xx'),
]